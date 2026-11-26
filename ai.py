
#!/usr/bin/env python3
"""
Local Coding MCP
- MCP tools for codebase exploration, debugging, editing and checks.
- Uses llama.cpp's OpenAI-compatible chat completions API.
- File operations are restricted to MCP_WORKSPACE.
"""

import ast
import difflib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# -------------------- CONFIGURATION --------------------

SERVER_NAME = "Local Coding MCP"
LLAMA_URL = os.getenv(
    "LLAMA_URL", "http://127.0.0.1:9931/v1/chat/completions"
)
MODEL_NAME = os.getenv("LLAMA_MODEL", "local-model")
WORKSPACE = Path(
    os.getenv("MCP_WORKSPACE", os.getcwd())
).expanduser().resolve()

MAX_FILE_CHARS = 40_000
MAX_MODEL_TOKENS = 8192
IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".next", "dist", "build", ".expo", ".gradle",
    ".idea", ".mypy_cache", ".pytest_cache",
}

mcp = FastMCP(SERVER_NAME)


# -------------------- CORE HELPERS --------------------

def safe_path(path: str) -> Path:
    """Resolve a relative path without allowing workspace escape."""
    candidate = (WORKSPACE / path).resolve()

    if not candidate.is_relative_to(WORKSPACE):
        raise ValueError("Access denied: path is outside the workspace.")

    return candidate


def read_source(path: str) -> tuple[Path, str]:
    file_path = safe_path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    if file_path.stat().st_size > MAX_FILE_CHARS * 4:
        raise ValueError("File is too large. Inspect smaller sections first.")

    content = file_path.read_text(encoding="utf-8")
    if len(content) > MAX_FILE_CHARS:
        raise ValueError(
            f"File exceeds {MAX_FILE_CHARS} characters. "
            "Inspect a smaller file or split the task."
        )

    return file_path, content


def ask_model(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Call the currently running llama.cpp server."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": min(max_tokens, MAX_MODEL_TOKENS),
        "stream": False,
    }

    request = urllib.request.Request(
        LLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach llama.cpp at {LLAMA_URL}. "
            "Start the llama-server and check its port. "
            f"Details: {exc}"
        ) from exc

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"No model response returned: {result}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content).strip()


def workspace_files(limit: int = 300) -> list[str]:
    found = []

    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)

        for name in sorted(files):
            path = Path(root) / name
            found.append(path.relative_to(WORKSPACE).as_posix())

            if len(found) >= limit:
                return found

    return found


# -------------------- WORKSPACE TOOLS --------------------

@mcp.tool()
def workspace_info() -> str:
    """Show the configured workspace and llama.cpp connection."""
    return (
        f"Workspace: {WORKSPACE}\n"
        f"Workspace exists: {WORKSPACE.is_dir()}\n"
        f"LLM endpoint: {LLAMA_URL}\n"
        f"Model identifier: {MODEL_NAME}\n"
        "File edits are confined to this workspace."
    )


@mcp.tool()
def list_files(path: str = ".", limit: int = 300) -> str:
    """List project files. Optionally restrict listing to a subdirectory."""
    base = safe_path(path)

    if not base.is_dir():
        return f"Not a directory: {path}"

    files = []
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)

        for name in sorted(names):
            full = Path(root) / name
            files.append(full.relative_to(WORKSPACE).as_posix())

            if len(files) >= min(max(limit, 1), 1000):
                return "\n".join(files)

    return "\n".join(files) if files else "No files found."


@mcp.tool()
def read_file(path: str) -> str:
    """Read a source file inside the workspace."""
    _, content = read_source(path)
    return content


@mcp.tool()
def search_code(query: str, path: str = ".") -> str:
    """Search source code for text, function names, classes or symbols."""
    base = safe_path(path)

    if not base.is_dir():
        raise ValueError("Search path must be a directory.")

    rg = subprocess.run(
        [
            "rg", "-n", "--hidden", "--smart-case",
            "--glob", "!.git/**",
            "--glob", "!node_modules/**",
            "--glob", "!.venv/**",
            "--glob", "!venv/**",
            "--glob", "!dist/**",
            "--glob", "!build/**",
            "--", query, str(base),
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = rg.stdout or rg.stderr or "No matches found."
    return output[:20000]


@mcp.tool()
def git_status() -> str:
    """Show current Git branch and working-tree changes."""
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout or result.stderr or "No Git output."


# -------------------- LOCAL AI TOOLS --------------------

@mcp.tool()
def ask_local_ai(question: str, context: str = "") -> str:
    """Ask the local GGUF model a coding or technical question."""
    return ask_model([
        {
            "role": "system",
            "content": (
                "You are a senior software engineer. Give accurate, "
                "practical answers. Explain bugs, assumptions and tradeoffs. "
                "Never claim you executed code unless a tool did so."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context[:20000]}",
        },
    ])


@mcp.tool()
def debug_file(path: str, error: str = "") -> str:
    """
    Analyze a source file for bugs, logic errors and likely causes
    of a supplied error. Does not modify the file.
    """
    _, source = read_source(path)

    return ask_model([
        {
            "role": "system",
            "content": (
                "You are a careful debugging engineer. Analyze the supplied "
                "source and error. Identify likely root causes, point to "
                "specific lines or functions, distinguish confirmed facts "
                "from hypotheses, and recommend a minimal fix. Do not "
                "pretend that you ran the program."
            ),
        },
        {
            "role": "user",
            "content": (
                f"File: {path}\n"
                f"Reported error:\n{error or '(not supplied)'}\n\n"
                f"Source code:\n```\n{source}\n```"
            ),
        },
    ])


@mcp.tool()
def fix_file(
    path: str,
    instructions: str,
    apply: bool = False,
) -> str:
    """
    Ask the local model for a complete corrected file.

    By default, returns a diff without changing anything.
    Set apply=True only when you explicitly want the proposed fix written.
    A timestamped backup is created before applying a fix.
    """
    file_path, original = read_source(path)

    answer = ask_model([
        {
            "role": "system",
            "content": (
                "You are a precise code repair engine. Return the COMPLETE "
                "replacement file, preserving existing functionality and "
                "making the smallest necessary changes. Do not omit code, "
                "use ellipses, or provide explanations outside the markers. "
                "Use exactly these markers on separate lines:\n"
                "<<<BEGIN_FILE>>>\n"
                "complete source here\n"
                "<<<END_FILE>>>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"File: {path}\n"
                f"Requested changes: {instructions}\n\n"
                f"Original source:\n{original}"
            ),
        },
    ], max_tokens=MAX_MODEL_TOKENS)

    match = re.search(
        r"<<<BEGIN_FILE>>>\s*\n(.*?)\n<<<END_FILE>>>",
        answer,
        flags=re.DOTALL,
    )

    if not match:
        return (
            "No complete replacement was detected. Nothing was changed. "
            "The model may have omitted the markers or run out of tokens. "
            "Try a smaller file or a more focused repair request.\n\n"
            f"Model response:\n{answer[:8000]}"
        )

    updated = match.group(1)

    if not updated.strip():
        return "The proposed file is empty. Nothing was changed."

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    ))

    if not diff:
        return "No changes proposed."

    if not apply:
        return (
            "PREVIEW ONLY — no files changed.\n"
            "Review this diff. Call fix_file again with apply=True "
            "only after deciding to apply it.\n\n"
            + diff[:18000]
        )

    # Basic syntax check for Python replacements before writing.
    if file_path.suffix == ".py":
        try:
            ast.parse(updated, filename=str(file_path))
        except SyntaxError as exc:
            return (
                f"Proposed Python code has a syntax error: {exc}. "
                "Nothing was changed.\n\n" + diff[:12000]
            )

    backup = file_path.with_name(
        file_path.name + f".backup-{int(time.time())}"
    )

    backup.write_bytes(file_path.read_bytes())
    file_path.write_text(updated, encoding="utf-8")

    return (
        f"Applied proposed fix to {path}\n"
        f"Backup: {backup.relative_to(WORKSPACE)}\n"
        "Review the diff and run the appropriate tests.\n\n"
        + diff[:18000]
    )


# -------------------- TEST AND VALIDATION TOOLS --------------------

@mcp.tool()
def run_check(check: str) -> str:
    """
    Run one predefined check in the workspace.
    Allowed: python_tests, python_compile, ruff, npm_test,
    npm_lint, typescript, git_diff_check.
    No arbitrary shell commands are accepted.
    """
    commands = {
        "python_tests": ["python", "-m", "pytest", "-q"],
        "python_compile": ["python", "-m", "compileall", "-q", "."],
        "ruff": ["ruff", "check", "."],
        "npm_test": ["npm", "test", "--", "--runInBand"],
        "npm_lint": ["npm", "run", "lint"],
        "typescript": ["npx", "tsc", "--noEmit"],
        "git_diff_check": ["git", "diff", "--check"],
    }

    if check not in commands:
        return (
            "Unknown check. Allowed checks: "
            + ", ".join(commands.keys())
        )

    if check.startswith("npm_") or check == "typescript":
        if not (WORKSPACE / "package.json").is_file():
            return "No package.json found in the workspace root."

    if check == "python_tests" and not (
        (WORKSPACE / "tests").exists()
        or list(WORKSPACE.glob("test_*.py"))
        or list(WORKSPACE.glob("*_test.py"))
    ):
        return "No obvious Python test files found in the workspace root."

    try:
        result = subprocess.run(
            commands[check],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        return f"Required executable is missing: {exc}"
    except subprocess.TimeoutExpired:
        return "Check timed out after 120 seconds."

    output = (result.stdout + "\n" + result.stderr).strip()

    return (
        f"Check: {check}\n"
        f"Exit code: {result.returncode}\n"
        f"Result: {'PASS' if result.returncode == 0 else 'FAIL'}\n\n"
        f"{output[:16000] or '(no output)'}"
    )


# -------------------- START MCP OVER STDIO --------------------

if __name__ == "__main__":
    print(
        f"{SERVER_NAME} starting; workspace={WORKSPACE}; "
        f"LLM={LLAMA_URL}",
        flush=True,
    )
    mcp.run(transport="stdio")
