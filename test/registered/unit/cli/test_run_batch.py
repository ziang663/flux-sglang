import json
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "python"))
sys.modules.setdefault("pybase64", types.SimpleNamespace())

from sglang.cli.run_batch import (
    OpenAIChatBatchSorter,
    _make_error_output,
    _parse_batch_request,
    _prepare_sorted_requests,
    _validate_chat_batch_request,
)


def test_parse_batch_request():
    req = _parse_batch_request(
        json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                },
            }
        ),
        line_no=1,
    )

    assert req.custom_id == "req-1"
    assert req.body["messages"][0]["content"] == "hello"


def test_validate_chat_batch_request_rejects_unsupported_url():
    req = _parse_batch_request(
        json.dumps(
            {
                "custom_id": "req-1",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {},
            }
        ),
        line_no=1,
    )

    with pytest.raises(ValueError, match="Only /v1/chat/completions is supported"):
        _validate_chat_batch_request(req)


def test_make_error_output_uses_openai_error_shape():
    output = _make_error_output(
        custom_id="req-1",
        request_id="rid-1",
        message="bad request",
        status_code=400,
    )

    assert output.custom_id == "req-1"
    assert output.response is not None
    assert output.response.status_code == 400
    assert output.response.body["object"] == "error"
    assert output.response.body["message"] == "bad request"


def test_sorter_groups_shared_prefix_and_shorter_lengths_first(tmp_path):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "b",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "messages": [{"role": "user", "content": "alpha prefix " + ("x" * 400)}],
                            "max_tokens": 64,
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "c",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "messages": [{"role": "user", "content": "beta prefix " + ("z" * 100)}],
                            "max_tokens": 16,
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "a",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "messages": [{"role": "user", "content": "alpha prefix " + ("y" * 120)}],
                            "max_tokens": 16,
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    artifacts = _prepare_sorted_requests(
        str(input_file),
        sort_enabled=True,
        start_prefix_tokens=64,
        end_suffix_tokens=8,
    )
    custom_ids = [task.request.custom_id for task in artifacts.tasks]

    assert custom_ids[0] == "a"
    assert custom_ids[1] == "b"
    assert custom_ids[2] == "c"


def test_sorter_collects_duplicate_custom_id_as_error(tmp_path):
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "dup",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"messages": [{"role": "user", "content": "hello"}]},
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "dup",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"messages": [{"role": "user", "content": "world"}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    sorter = OpenAIChatBatchSorter()
    tasks = sorter.load_tasks(str(input_file))

    assert [task.request.custom_id for task in tasks] == ["dup"]
    assert len(sorter.invalid_tasks) == 1
    assert "Duplicate custom_id" in sorter.invalid_tasks[0].response.body["message"]
