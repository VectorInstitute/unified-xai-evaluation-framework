"""Ingest HAL TAU-bench traces into Docent as episode-level AgentRuns."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message


TRACE_PATH = (
    "./data/Traces/Taubenchairline/Taubenchairline/taubench_airline_1743994890_UPLOAD.json"
)


def safe_content(value: Any) -> str:
    """Convert value to string, handling None safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


# TODO: Refactor to reduce complexity
def load_hal_weave_runs(data: Dict[str, Any]) -> List[AgentRun]:  # noqa: PLR0912, PLR0915
    """
    Load HAL TAU-bench runs from Weave trace data.

    Build ONE AgentRun per weave_task_id (episode-level),
    including:
      - user / system messages
      - assistant messages
      - tool calls
      - tool responses
      - tool errors
      - span-level exceptions
    """
    config = data.get("config", {})
    benchmark_name = config.get("benchmark_name", "taubench_airline")
    agent_name = config.get("agent_name", "unknown_agent")

    spans = data.get("raw_logging_results", [])

    spans_by_task: Dict[str, List[dict]] = defaultdict(list)

    for span in spans:
        task_id = span.get("weave_task_id")
        if task_id is not None:
            spans_by_task[str(task_id)].append(span)

    agent_runs: List[AgentRun] = []
    for task_id, task_spans in spans_by_task.items():
        task_spans.sort(key=lambda s: (s.get("started_at", ""), s.get("ended_at", "")))

        messages = []
        seen = set()

        for span in task_spans:
            inputs = span.get("inputs", {})
            output = span.get("output", {})

            for msg in inputs.get("messages", []):
                role = msg.get("role", "user")
                content = safe_content(msg.get("content"))
                key = (role, content)

                if content.strip() and key not in seen:
                    seen.add(key)
                    messages.append(
                        parse_chat_message(
                            {
                                "role": role,
                                "content": content,
                                "metadata": {
                                    "source": "input",
                                    "op_name": span.get("op_name"),
                                },
                            }
                        )
                    )

            for choice in output.get("choices", []):
                assistant_msg = choice.get("message", {})

                content = safe_content(assistant_msg.get("content"))
                key = ("assistant", content)

                if content.strip() and key not in seen:
                    seen.add(key)
                    messages.append(
                        parse_chat_message(
                            {
                                "role": "assistant",
                                "content": content,
                                "metadata": {
                                    "source": "assistant",
                                    "op_name": span.get("op_name"),
                                },
                            }
                        )
                    )

                for tool_call in assistant_msg.get("tool_calls", []) or []:
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", "")
                    call_repr = f"{tool_name}({tool_args})"
                    key = ("tool_call", call_repr)

                    if call_repr.strip() and key not in seen:
                        seen.add(key)
                        messages.append(
                            parse_chat_message(
                                {
                                    "role": "tool",
                                    "content": f"[TOOL CALL] {call_repr}",
                                    "metadata": {
                                        "tool_name": tool_name,
                                        "type": "call",
                                    },
                                }
                            )
                        )

            for msg in inputs.get("messages", []):
                if msg.get("role") == "tool":
                    content = safe_content(msg.get("content"))
                    is_error = content.lower().startswith("error")
                    key = ("tool_response", content)

                    if content.strip() and key not in seen:
                        seen.add(key)
                        messages.append(
                            parse_chat_message(
                                {
                                    "role": "tool",
                                    "content": (
                                        f"[TOOL ERROR] {content}"
                                        if is_error
                                        else f"[TOOL RESPONSE] {content}"
                                    ),
                                    "metadata": {
                                        "tool_name": msg.get("name"),
                                        "type": "response",
                                        "is_error": is_error,
                                    },
                                }
                            )
                        )

            exception = span.get("exception")
            if exception:
                content = safe_content(exception)
                key = ("exception", content)

                if content.strip() and key not in seen:
                    seen.add(key)
                    messages.append(
                        parse_chat_message(
                            {
                                "role": "error",
                                "content": f"[EXCEPTION] {content}",
                                "metadata": {
                                    "source": "span_exception",
                                    "op_name": span.get("op_name"),
                                },
                            }
                        )
                    )

        if not messages:
            continue
        transcript = Transcript(
            messages=messages,
            metadata={
                "task_id": task_id,
                "benchmark": benchmark_name,
            },
        )

        agent_runs.append(
            AgentRun(
                transcripts=[transcript],
                metadata={
                    "task_id": task_id,
                    "benchmark": benchmark_name,
                    "agent_name": agent_name,
                    "total_cost": data.get("total_cost"),
                    "total_usage": data.get("total_usage"),
                },
            )
        )

    return agent_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest HAL TAU-bench traces into Docent"
    )
    parser.add_argument(
        "--trace-path", type=str, default=TRACE_PATH, help="Path to the trace JSON file"
    )

    args = parser.parse_args()

    with open(args.trace_path, "r") as f:
        log = json.load(f)

    client = Docent(api_key=os.getenv("DOCENT_API_KEY"))

    collection_id = client.create_collection(
        name="HAL TAU-bench Airline (Episode-level, with errors)",
        description="One AgentRun per task with tools, failures, and exceptions",
    )

    print(f"Created collection: {collection_id}")

    agent_runs = load_hal_weave_runs(log)
    print(f"Ingesting {len(agent_runs)} agent runs...")

    client.add_agent_runs(collection_id, agent_runs)

    print("✅ Ingestion complete")
