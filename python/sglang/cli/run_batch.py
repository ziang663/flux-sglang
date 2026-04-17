from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


class BatchRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom_id: str
    method: str
    url: str
    body: dict[str, Any]


class BatchResponseData(BaseModel):
    status_code: int
    request_id: str
    body: dict[str, Any]


class BatchRequestOutput(BaseModel):
    id: str
    custom_id: str
    response: Optional[BatchResponseData]
    error: Optional[dict[str, Any]]


@dataclass
class RequestTask:
    line_no: int
    request: BatchRequestInput
    prompt_text: str
    prompt_tokens_est: int
    output_tokens_est: int
    original_index: int


@dataclass
class SortArtifacts:
    tasks: list[RequestTask] = field(default_factory=list)
    outputs: list[BatchRequestOutput] = field(default_factory=list)


class BaseTaskSorter(ABC):
    def __init__(self):
        self.tasks: list[Any] = []
        self.sorted_tasks: list[Any] = []

    @abstractmethod
    def read_dataset(self, input_uri: str, **kwargs) -> tuple[list[Any], list[Any]]:
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(self, task: Any, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def estimate_affinity(self, task: Any, length: int, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def write_output(self, sorted_tasks: list[Any], output_uri: str, **kwargs) -> None:
        raise NotImplementedError

    def load_tasks(self, input_uri: str, **kwargs) -> list[Any]:
        self.tasks, invalid_tasks = self.read_dataset(input_uri, **kwargs)
        self.invalid_tasks = invalid_tasks
        return self.tasks

    def sort_tasks(self, **kwargs) -> list[Any]:
        if not self.tasks:
            self.sorted_tasks = []
            return self.sorted_tasks

        start_prefix_tokens = kwargs.get("start_prefix_tokens", 4096)
        end_suffix_tokens = kwargs.get("end_suffix_tokens", 128)

        s, e = start_prefix_tokens, end_suffix_tokens
        level_des = []
        current = s
        while current > e:
            level_des.append(current)
            current = current // 2
            if current == level_des[-1]:
                break
        if e not in level_des:
            level_des.append(e)
        level_des = sorted(set(level_des), reverse=True)
        level_asc = sorted(level_des)

        items = []
        for i, task in enumerate(self.tasks):
            score = self.estimate_cost(task, **kwargs)
            sigs = {level: self.estimate_affinity(task, level, **kwargs) for level in level_des}
            items.append((task, score, sigs, i))

        counts: dict[int, dict[Any, int]] = {level: {} for level in level_asc}
        for _, _, sigs, _ in items:
            for level in level_asc:
                sig = sigs[level]
                counts[level][sig] = counts[level].get(sig, 0) + 1

        def _tree_key(item):
            task, score, sigs, idx = item
            key_parts: list[Any] = []
            for level in level_asc:
                sig = sigs[level]
                key_parts.append(-counts[level][sig])
                key_parts.append(sig)
            # Prefer grouping by total cost to keep lengths similar within local neighborhoods.
            key_parts.append(score)
            key_parts.append(getattr(task, "prompt_tokens_est", 0))
            key_parts.append(idx)
            return tuple(key_parts)

        items.sort(key=_tree_key)
        self.sorted_tasks = [task for task, _, _, _ in items]
        return self.sorted_tasks


class OpenAIChatBatchSorter(BaseTaskSorter):
    def read_dataset(
        self,
        input_uri: str,
        **kwargs,
    ) -> tuple[list[RequestTask], list[BatchRequestOutput]]:
        tasks: list[RequestTask] = []
        outputs: list[BatchRequestOutput] = []
        seen_custom_ids: set[str] = set()

        with open(input_uri, encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                request_id = uuid.uuid4().hex
                try:
                    req = _parse_batch_request(stripped, line_no)
                    _validate_chat_batch_request(req)
                except ValueError as exc:
                    outputs.append(
                        _make_error_output(
                            custom_id=f"line-{line_no}",
                            request_id=request_id,
                            message=str(exc),
                            status_code=400,
                        )
                    )
                    continue

                if req.custom_id in seen_custom_ids:
                    outputs.append(
                        _make_error_output(
                            custom_id=req.custom_id,
                            request_id=request_id,
                            message=f"Duplicate custom_id '{req.custom_id}'.",
                            status_code=400,
                            param="custom_id",
                        )
                    )
                    continue
                seen_custom_ids.add(req.custom_id)

                prompt_text = _extract_prompt_text(req.body)
                tasks.append(
                    RequestTask(
                        line_no=line_no,
                        request=req,
                        prompt_text=prompt_text,
                        prompt_tokens_est=_estimate_tokens(prompt_text),
                        output_tokens_est=_extract_output_tokens_est(req.body),
                        original_index=len(tasks),
                    )
                )

        return tasks, outputs

    def estimate_cost(self, task: RequestTask, **kwargs) -> int:
        return task.prompt_tokens_est + task.output_tokens_est

    def estimate_affinity(self, task: RequestTask, length: int, **kwargs) -> str:
        normalized = _normalize_prompt(task.prompt_text)
        # Use coarse windows on lower levels and progressively finer windows on higher levels.
        # This lets nearby prompts cluster even when they diverge shortly after a shared prefix.
        prefix = normalized[: max(length, 16)]
        if not prefix:
            return "empty"
        return hashlib.sha1(prefix.encode("utf-8")).hexdigest()

    def write_output(self, sorted_tasks: list[RequestTask], output_uri: str, **kwargs) -> None:
        raise NotImplementedError("Output writing is handled by run_batch().")


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def _estimate_tokens(text: str) -> int:
    normalized = _normalize_prompt(text)
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _extract_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(message, dict):
        content = message.get("content")
        return _extract_message_text(content)
    return ""


def _extract_prompt_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        segments = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "unknown")
            content = _extract_message_text(message)
            if content:
                segments.append(f"{role}: {content}")
        if segments:
            return "\n".join(segments)
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return ""


def _extract_output_tokens_est(body: dict[str, Any]) -> int:
    value = body.get("max_completion_tokens", body.get("max_tokens", 16))
    if isinstance(value, int):
        return max(0, value)
    return 16


def _prepare_sorted_requests(
    input_file: str,
    sort_enabled: bool,
    start_prefix_tokens: int,
    end_suffix_tokens: int,
) -> SortArtifacts:
    sorter = OpenAIChatBatchSorter()
    tasks = sorter.load_tasks(input_file)
    if sort_enabled:
        tasks = sorter.sort_tasks(
            start_prefix_tokens=start_prefix_tokens,
            end_suffix_tokens=end_suffix_tokens,
        )
    return SortArtifacts(tasks=tasks, outputs=list(getattr(sorter, "invalid_tasks", [])))


def _parse_args(extra_argv: list[str]) -> argparse.Namespace:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(
        description="Run offline batch prompts from a JSONL file.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--disable-request-sort",
        action="store_true",
        help="Disable asynchronous request reordering during model initialization.",
    )
    parser.add_argument(
        "--sort-start-prefix-tokens",
        type=int,
        default=4096,
        help="Largest prefix level used by the offline request sorter.",
    )
    parser.add_argument(
        "--sort-end-suffix-tokens",
        type=int,
        default=128,
        help="Smallest prefix level used by the offline request sorter.",
    )
    ServerArgs.add_cli_args(parser)
    return parser.parse_args(extra_argv)


def _make_error_output(
    custom_id: str,
    request_id: str,
    message: str,
    status_code: int = 400,
    err_type: str = "BadRequestError",
    param: str | None = None,
) -> BatchRequestOutput:
    from sglang.srt.entrypoints.openai.protocol import ErrorResponse

    error = ErrorResponse(
        message=message,
        type=err_type,
        code=status_code,
        param=param,
    )
    return BatchRequestOutput(
        id=f"batch-{request_id}",
        custom_id=custom_id,
        response=BatchResponseData(
            status_code=status_code,
            request_id=request_id,
            body=error.model_dump(),
        ),
        error=None,
    )


def _parse_batch_request(line: str, line_no: int) -> BatchRequestInput:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc

    try:
        return BatchRequestInput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Line {line_no}: invalid batch request: {exc}") from exc


def _validate_chat_batch_request(req: BatchRequestInput) -> None:
    if req.method.upper() != "POST":
        raise ValueError(f"Unsupported method '{req.method}'. Only POST is supported.")
    if req.url != "/v1/chat/completions":
        raise ValueError(
            f"Unsupported url '{req.url}'. Only /v1/chat/completions is supported."
        )


def _response_to_output(
    custom_id: str,
    request_id: str,
    response: dict[str, Any],
    status_code: int = 200,
) -> BatchRequestOutput:
    body_request_id = response.get("id", request_id)
    return BatchRequestOutput(
        id=f"batch-{body_request_id}",
        custom_id=custom_id,
        response=BatchResponseData(
            status_code=status_code,
            request_id=body_request_id,
            body=response,
        ),
        error=None,
    )


def _process_chat_request(
    req: BatchRequestInput,
    engine: Engine,
    serving_chat: OpenAIServingChat,
    default_model: str,
) -> BatchRequestOutput:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    request_id = uuid.uuid4().hex
    _validate_chat_batch_request(req)

    body = dict(req.body)
    body.setdefault("model", default_model)
    body["stream"] = False
    body.setdefault("rid", f"batch-{req.custom_id}-{request_id}")

    try:
        chat_request = ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        return _make_error_output(
            custom_id=req.custom_id,
            request_id=request_id,
            message=str(exc),
            status_code=400,
        )

    error_msg = serving_chat._validate_request(chat_request)
    if error_msg:
        return _make_error_output(
            custom_id=req.custom_id,
            request_id=request_id,
            message=error_msg,
            status_code=400,
        )

    try:
        adapted_request, processed_request = serving_chat._convert_to_internal_request(
            chat_request
        )
        ret = serving_chat.tokenizer_manager.generate_request(
            adapted_request, None
        ).__anext__()
        result = engine.loop.run_until_complete(ret)
        if not isinstance(result, list):
            result = [result]
        response = serving_chat._build_chat_response(
            processed_request,
            result,
            int(time.time()),
        )
    except ValueError as exc:
        return _make_error_output(
            custom_id=req.custom_id,
            request_id=request_id,
            message=str(exc),
            status_code=400,
        )
    except Exception as exc:
        return _make_error_output(
            custom_id=req.custom_id,
            request_id=request_id,
            message=f"Internal server error: {exc}",
            status_code=500,
            err_type="InternalServerError",
        )

    if isinstance(response, ORJSONResponse):
        body_dict = json.loads(response.body)
        response_id = body_dict.get("id", request_id)
        return BatchRequestOutput(
            id=f"batch-{response_id}",
            custom_id=req.custom_id,
            response=BatchResponseData(
                status_code=response.status_code,
                request_id=response_id,
                body=body_dict,
            ),
            error=None,
        )

    return _response_to_output(
        custom_id=req.custom_id,
        request_id=request_id,
        response=response.model_dump(mode="json", exclude_none=True),
    )


def run_batch(args, extra_argv):
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.server_args import ServerArgs

    parsed_args = _parse_args(extra_argv)
    server_args = ServerArgs.from_cli_args(parsed_args)
    default_model = server_args.served_model_name or server_args.model_path
    sort_enabled = not parsed_args.disable_request_sort

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        sort_future = executor.submit(
            _prepare_sorted_requests,
            parsed_args.input_file,
            sort_enabled,
            parsed_args.sort_start_prefix_tokens,
            parsed_args.sort_end_suffix_tokens,
        )

        engine = Engine(**dataclasses.asdict(server_args))
        serving_chat = OpenAIServingChat(
            engine.tokenizer_manager, engine.template_manager
        )
        artifacts = sort_future.result()
        outputs = list(artifacts.outputs)

    try:
        for task in artifacts.tasks:
            outputs.append(
                _process_chat_request(
                    req=task.request,
                    engine=engine,
                    serving_chat=serving_chat,
                    default_model=default_model,
                )
            )
    finally:
        engine.shutdown()

    with open(parsed_args.output_file, "w", encoding="utf-8") as fout:
        for output in outputs:
            fout.write(
                json.dumps(output.model_dump(mode="json", exclude_none=False)) + "\n"
            )
