
# tools/print_project_tree.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ---- defaults: ignore noisy dirs/files ----
DEFAULT_IGNORE_DIRS = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".DS_Store",
    "node_modules",
    # venvs
    ".venv", ".venv311", "venv", "env",
    # python packaging noise
    "site-packages", "dist", "build",
    # model/checkpoints
    "checkpoints", "checkpoint", "runs",
    # data too large (keep data/ itself, but skip heavy internals if you want)
    ".ipynb_checkpoints",
}

DEFAULT_IGNORE_EXT = {
    ".pyc", ".pyo", ".pyd",
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp4", ".mov",
    ".zip", ".tar", ".gz", ".7z",
    ".safetensors", ".bin",
}

DEFAULT_IGNORE_FILES = {
    ".DS_Store",
}


def should_ignore(path: Path, ignore_dirs: set[str], ignore_ext: set[str], ignore_files: set[str]) -> bool:
    name = path.name

    if path.is_dir():
        if name in ignore_dirs:
            return True
        # also ignore hidden dirs by default (optional)
        # if name.startswith("."): return True
        return False

    # file
    if name in ignore_files:
        return True
    if path.suffix.lower() in ignore_ext:
        return True
    return False


def walk(root: Path, prefix: str, depth: int, max_depth: int,
         ignore_dirs: set[str], ignore_ext: set[str], ignore_files: set[str]) -> None:
    if depth > max_depth:
        return

    try:
        entries = sorted(list(root.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    # filter
    entries = [p for p in entries if not should_ignore(p, ignore_dirs, ignore_ext, ignore_files)]

    for i, p in enumerate(entries):
        is_last = (i == len(entries) - 1)
        branch = "└── " if is_last else "├── "
        print(prefix + branch + p.name + ("/" if p.is_dir() else ""))

        if p.is_dir():
            walk(
                p,
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                max_depth,
                ignore_dirs,
                ignore_ext,
                ignore_files,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (default: .)")
    ap.add_argument("--max_depth", type=int, default=4, help="max depth to print (default: 4)")
    ap.add_argument("--out", default="docs/project_tree.txt", help="output file path")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ignore_dirs = set(DEFAULT_IGNORE_DIRS)
    ignore_ext = set(DEFAULT_IGNORE_EXT)
    ignore_files = set(DEFAULT_IGNORE_FILES)

    # write to file
    lines = []
    lines.append(f"{root.name}/")

    # capture stdout-like output by building lines
    def _capture_walk(r: Path, prefix: str, depth: int):
        if depth > args.max_depth:
            return
        try:
            entries = sorted(list(r.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        entries2 = [p for p in entries if not should_ignore(p, ignore_dirs, ignore_ext, ignore_files)]
        for i, p in enumerate(entries2):
            is_last = (i == len(entries2) - 1)
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + p.name + ("/" if p.is_dir() else ""))
            if p.is_dir():
                _capture_walk(p, prefix + ("    " if is_last else "│   "), depth + 1)

    _capture_walk(root, "", 1)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote -> {out_path}")
    print("[preview]")
    print("\n".join(lines[:80]))
    if len(lines) > 80:
        print(f"... ({len(lines)} lines total)")


if __name__ == "__main__":
    main()