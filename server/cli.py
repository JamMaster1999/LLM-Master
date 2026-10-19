"""Drives one conversation with the backend from the terminal.

    uv run python -m server.cli [--model claude-sonnet-5-5] [--conversation ID] [--files DIR] [--url URL]

Type a message and press enter. `/attach PATH` adds a file to the next message, `/delete` deletes the
conversation, and `/quit` leaves. Ctrl-C stops a turn. Files the agent sends land in --files. Without
--url, the CLI asks Modal for the deployed app's address in the dev environment. MODAL_PROXY_TOKEN,
a proxy token written `wk-….ws-…`, signs the requests.
"""
import argparse
import base64
import json
import mimetypes
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import modal

from server.app import MODELS

DIM, PLAIN = "\033[2m", "\033[0m"


def chunks(client: httpx.Client, url: str, body: dict) -> Iterator[dict]:
    """The chunks of one request's stream."""
    with client.stream("POST", f"{url}/chat", json=body) as response:
        if response.is_error:
            raise SystemExit(f"{response.status_code}: {response.read().decode()}")
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                yield json.loads(line.removeprefix("data: "))


def attached(path: Path) -> dict:
    """An upload as the AI SDK sends it: a file part holding a data URL."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {"type": "file", "mediaType": mime, "filename": path.name,
            "url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"}


def show(chunk: dict, files: Path) -> None:
    match chunk["type"]:
        case "text-delta":
            print(chunk["delta"], end="", flush=True)
        case "reasoning-delta":
            print(DIM + chunk["delta"] + PLAIN, end="", flush=True)
        case "text-end" | "reasoning-end":
            print()
        case "tool-input-available":
            print(f"{DIM}→ {chunk['toolName']} {json.dumps(chunk['input'], ensure_ascii=False)[:300]}{PLAIN}")
        case "tool-output-available":
            print(f"{DIM}← {chunk['output'][:300]}{PLAIN}")
        case "tool-output-error":
            print(f"✗ {chunk['errorText'][:300]}")
        case "source-url":
            print(f"[{chunk['sourceId']}] {chunk.get('title', '')} {chunk['url']}")
        case "file":
            files.mkdir(parents=True, exist_ok=True)
            path = files / chunk["providerMetadata"]["agent"]["filename"]
            path.write_bytes(base64.b64decode(chunk["url"].partition(",")[2]))
            print(f"Saved {path.resolve()}")
        case "error":
            print(f"Error: {chunk['errorText']}")


def answered(calls: dict[str, dict]) -> dict:
    """Asks ask_user's questions here, and returns the assistant message that carries the answers, as
    the AI SDK sends it. An empty reply declines."""
    parts = []
    for call_id, arguments in calls.items():
        answers: dict[str, str] | None = {}
        for question in arguments["questions"]:
            print(question["question"])
            for number, option in enumerate(question["options"], 1):
                print(f"  {number}. {option['label']}: {option['description']}")
            reply = input("answer (numbers, or your own words)> ").strip()
            if not reply:
                answers = None
                break
            picked = [question["options"][int(n) - 1]["label"] for n in reply.replace(",", " ").split()
                      if n.isdigit() and 0 < int(n) <= len(question["options"])]
            answers[question["question"]] = ", ".join(picked) or reply
        parts.append({"type": "tool-ask_user", "toolCallId": call_id, "state": "output-available", "output": answers})
    return {"role": "assistant", "parts": parts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default="claude-sonnet-5-5")
    parser.add_argument("--conversation", default=str(uuid.uuid4()))
    parser.add_argument("--files", type=Path, default=Path("files"))
    parser.add_argument("--url", help="the backend's address; found on Modal when left out")
    parser.add_argument("--app", default="llm-master-server")
    args = parser.parse_args()
    url = args.url or modal.Function.from_name(args.app, "web", environment_name="dev").get_web_url()
    assert url, "web() is a web endpoint, so it has an address"
    token = os.environ.get("MODAL_PROXY_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"Conversation {args.conversation} with {args.model}. /attach PATH, /delete, /quit.")
    uploads: list[dict] = []
    message = None
    with httpx.Client(headers=headers, timeout=httpx.Timeout(30, read=None)) as client:
        while True:
            if message is None:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if line == "/quit":
                    break
                if line == "/delete":
                    client.delete(f"{url}/conversations/{args.conversation}").raise_for_status()
                    print("Deleted.")
                    break
                if line.startswith("/attach "):
                    try:
                        uploads.append(attached(Path(line.removeprefix("/attach ").strip()).expanduser()))
                    except OSError as error:
                        print(error)
                    continue
                message, uploads = {"role": "user", "parts": [*uploads, {"type": "text", "text": line}]}, []
            asked = {}
            try:
                for chunk in chunks(client, url, {"id": args.conversation, "model": args.model, "message": message}):
                    show(chunk, args.files)
                    if chunk["type"] == "tool-input-available" and chunk["toolName"] == "ask_user":
                        asked[chunk["toolCallId"]] = chunk["input"]
            except KeyboardInterrupt:
                print("\n[stopped]")
                asked = {}
            message = answered(asked) if asked else None


if __name__ == "__main__":
    main()
