"""
用法：

1. 按默认路径导入单个 txt/md：
    python .\Clara\import_markdown_chat.py .\Clara\clara.txt --assistant-model Clara --force

2. 指定输出数据库路径：
    python .\Clara\import_markdown_chat.py .\Clara\clara.txt .\somewhere\db.sqlite --assistant-model Clara --force

3. 可选参数：
    --title 覆盖标题
    --start-time 指定首条消息时间，格式例如 2026-03-08T19:00:45+08:00
    --step-seconds 指定相邻消息的合成时间间隔
    --force 覆盖已有输出数据库

说明：
- 第一行作为标题。
- ### content 按顺序交替映射为 user / assistant。
- ### thinking 绑定到它前一条 assistant 消息。
- 默认输出到 ollamaDB/imports/<来源路径摘要>/db.sqlite。
"""

import argparse
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


DEFAULT_ASSISTANT_MODEL = "reconstructed-import"
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_IMPORT_ROOT = Path(__file__).with_name("imports")
PATH_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        device_id TEXT NOT NULL DEFAULT '',
        has_completed_first_run BOOLEAN NOT NULL DEFAULT 0,
        expose BOOLEAN NOT NULL DEFAULT 0,
        survey BOOLEAN NOT NULL DEFAULT TRUE,
        browser BOOLEAN NOT NULL DEFAULT 0,
        models TEXT NOT NULL DEFAULT '',
        agent BOOLEAN NOT NULL DEFAULT 0,
        tools BOOLEAN NOT NULL DEFAULT 0,
        working_dir TEXT NOT NULL DEFAULT '',
        context_length INTEGER NOT NULL DEFAULT 4096,
        window_width INTEGER NOT NULL DEFAULT 0,
        window_height INTEGER NOT NULL DEFAULT 0,
        config_migrated BOOLEAN NOT NULL DEFAULT 0,
        airplane_mode BOOLEAN NOT NULL DEFAULT 0,
        turbo_enabled BOOLEAN NOT NULL DEFAULT 0,
        websearch_enabled BOOLEAN NOT NULL DEFAULT 0,
        selected_model TEXT NOT NULL DEFAULT '',
        sidebar_open BOOLEAN NOT NULL DEFAULT 0,
        think_enabled BOOLEAN NOT NULL DEFAULT 0,
        think_level TEXT NOT NULL DEFAULT '',
        remote TEXT NOT NULL DEFAULT '',
        schema_version INTEGER NOT NULL DEFAULT 12,
        cloud_setting_migrated BOOLEAN NOT NULL DEFAULT 0,
        auto_update_enabled BOOLEAN NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE chats (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        browser_state TEXT
    )
    """,
    """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        thinking TEXT NOT NULL DEFAULT '',
        stream BOOLEAN NOT NULL DEFAULT 0,
        model_name TEXT,
        model_cloud BOOLEAN,
        model_ollama_host BOOLEAN,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        thinking_time_start TIMESTAMP,
        thinking_time_end TIMESTAMP,
        tool_result TEXT,
        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX idx_messages_chat_id ON messages(chat_id)",
    """
    CREATE TABLE tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        function_name TEXT NOT NULL,
        function_arguments TEXT NOT NULL,
        function_result TEXT,
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX idx_tool_calls_message_id ON tool_calls(message_id)",
    """
    CREATE TABLE attachments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        data BLOB NOT NULL,
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX idx_attachments_message_id ON attachments(message_id)",
    """
    CREATE TABLE users (
        name TEXT NOT NULL DEFAULT '',
        email TEXT NOT NULL DEFAULT '',
        plan TEXT NOT NULL DEFAULT '',
        cached_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


@dataclass
class ParsedMessage:
    role: str
    content: str
    thinking: str = ""


@dataclass
class ParsedChat:
    title: str
    messages: List[ParsedMessage]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Ollama-style SQLite chat database from a markdown/txt transcript."
    )
    parser.add_argument("input_file", type=Path, help="Markdown/txt transcript to import.")
    parser.add_argument(
        "output_db",
        nargs="?",
        type=Path,
        help="Target db.sqlite path to create. Defaults to ollamaDB/imports/<input_stem>/db.sqlite.",
    )
    parser.add_argument(
        "--assistant-model",
        default=DEFAULT_ASSISTANT_MODEL,
        help="Model name written into assistant messages.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Override the chat title. Defaults to the first markdown heading.",
    )
    parser.add_argument(
        "--start-time",
        default="",
        help="ISO datetime for the first message. Defaults to current local time.",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=8,
        help="Synthetic seconds between messages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_db if it already exists.",
    )
    return parser.parse_args()


def strip_outer_blank_lines(text: str) -> str:
    return text.strip("\n\r \t")


def parse_transcript(input_file: Path, override_title: str) -> ParsedChat:
    raw_text = input_file.read_text(encoding="utf-8-sig")
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        raise ValueError("Input file is empty.")

    first_line = lines[0].strip()
    title = override_title.strip() or first_line.lstrip("#").strip()
    if not title:
        raise ValueError("Missing title on the first line.")

    blocks = []
    current_kind: Optional[str] = None
    current_lines: List[str] = []

    for line in lines[1:]:
        stripped = line.strip()
        if stripped in {"### content", "### thinking"}:
            if current_kind is not None:
                blocks.append((current_kind, strip_outer_blank_lines("\n".join(current_lines))))
            current_kind = stripped.removeprefix("### ")
            current_lines = []
            continue
        if current_kind is not None:
            current_lines.append(line)

    if current_kind is not None:
        blocks.append((current_kind, strip_outer_blank_lines("\n".join(current_lines))))

    if not blocks:
        raise ValueError("No ### content blocks found.")

    messages: List[ParsedMessage] = []
    next_role = "user"

    for kind, block_text in blocks:
        if kind == "content":
            messages.append(ParsedMessage(role=next_role, content=block_text))
            next_role = "assistant" if next_role == "user" else "user"
            continue
        if not messages or messages[-1].role != "assistant":
            raise ValueError("A ### thinking block must follow an assistant ### content block.")
        previous_thinking = messages[-1].thinking
        messages[-1].thinking = block_text if not previous_thinking else previous_thinking + "\n\n" + block_text

    if not messages:
        raise ValueError("No messages were parsed from the transcript.")

    return ParsedChat(title=title, messages=messages)


def parse_start_time(value: str) -> datetime:
    if not value.strip():
        return datetime.now().astimezone()
    return datetime.fromisoformat(value)


def ensure_parent_dir(output_db: Path) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)


def sanitize_path_part(value: str) -> str:
    cleaned = PATH_SAFE_CHARS.sub("_", value.strip())
    return cleaned.strip("._-") or "item"


def derive_default_output_db(input_file: Path) -> Path:
    parent_parts = [sanitize_path_part(part) for part in input_file.parent.parts if part not in {".", ""}]
    base_name = "__".join(parent_parts + [sanitize_path_part(input_file.stem)])
    return DEFAULT_IMPORT_ROOT / base_name / "db.sqlite"


def initialize_database(connection: sqlite3.Connection, assistant_model: str) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        """
        INSERT INTO settings (id, has_completed_first_run, selected_model, think_enabled, context_length, schema_version)
        VALUES (1, 1, ?, 1, ?, 12)
        """,
        (assistant_model, DEFAULT_CONTEXT_LENGTH),
    )
    connection.execute("INSERT INTO users DEFAULT VALUES")


def format_timestamp(value: datetime) -> str:
    return value.isoformat(sep=" ", timespec="microseconds")


def write_chat(
    connection: sqlite3.Connection,
    chat: ParsedChat,
    assistant_model: str,
    start_time: datetime,
    step_seconds: int,
) -> str:
    chat_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO chats (id, title, created_at, browser_state) VALUES (?, ?, ?, NULL)",
        (chat_id, chat.title, format_timestamp(start_time)),
    )

    current_time = start_time
    for message in chat.messages:
        created_at = current_time
        updated_at = current_time
        model_name = None
        thinking_time_start = None
        thinking_time_end = None

        if message.role == "assistant":
            model_name = assistant_model
            if message.thinking:
                thinking_time_start = created_at
                thinking_time_end = created_at + timedelta(seconds=max(1, step_seconds // 2))
                updated_at = thinking_time_end

        connection.execute(
            """
            INSERT INTO messages (
                chat_id, role, content, thinking, stream, model_name,
                model_cloud, model_ollama_host, created_at, updated_at,
                thinking_time_start, thinking_time_end, tool_result
            )
            VALUES (?, ?, ?, ?, 0, ?, NULL, NULL, ?, ?, ?, ?, '')
            """,
            (
                chat_id,
                message.role,
                message.content,
                message.thinking,
                model_name,
                format_timestamp(created_at),
                format_timestamp(updated_at),
                format_timestamp(thinking_time_start) if thinking_time_start else None,
                format_timestamp(thinking_time_end) if thinking_time_end else None,
            ),
        )
        current_time = current_time + timedelta(seconds=step_seconds)

    return chat_id


def build_database(args: argparse.Namespace) -> tuple[Path, str]:
    parsed_chat = parse_transcript(args.input_file, args.title)
    output_db = args.output_db or derive_default_output_db(args.input_file)

    if output_db.exists():
        if not args.force:
            raise FileExistsError(f"Output database already exists: {output_db}")
        output_db.unlink()

    ensure_parent_dir(output_db)

    with sqlite3.connect(output_db) as connection:
        initialize_database(connection, args.assistant_model)
        chat_id = write_chat(
            connection=connection,
            chat=parsed_chat,
            assistant_model=args.assistant_model,
            start_time=parse_start_time(args.start_time),
            step_seconds=args.step_seconds,
        )
        connection.commit()

    return output_db, chat_id


def main() -> int:
    args = parse_args()
    output_db, chat_id = build_database(args)
    print(f"Created database: {output_db}")
    print(f"Imported chat_id: {chat_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())