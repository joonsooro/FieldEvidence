#!/usr/bin/env python3
"""
Verify that the current working tree only changed comments (no code tokens).

Checks:
- Python: AST equality before/after (using HEAD as baseline).
- JS/TS/Shell/YAML: naive comment-stripping + byte-equality as a heuristic.
"""

from __future__ import annotations
import subprocess, sys, re, tokenize, io, os, json, pathlib, ast

ROOT = pathlib.Path(__file__).resolve().parents[1]

INCLUDE_EXT = {".py",".js",".ts",".tsx",".jsx",".sh",".bash",".zsh",".yaml",".yml",".toml"}
EXCLUDE_DIRS = {".git",".venv","out","__pycache__"}

def git_ls_modified():
    out = subprocess.check_output(["git","diff","--name-only","HEAD"], cwd=ROOT).decode().splitlines()
    return [p for p in out if p]

def should_check(path: pathlib.Path)->bool:
    if not path.exists(): return False
    if any(part in EXCLUDE_DIRS for part in path.parts): return False
    return path.suffix in INCLUDE_EXT

def strip_python_comments(src: str)->str:
    # For heuristic compare only (we use AST for python anyway)
    out = []
    tokgen = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in tokgen:
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok.string)
    return "".join(out)

JS_BLOCK = re.compile(r"/\*.*?\*/", re.S)
JS_LINE  = re.compile(r"//[^\n]*")
SHELL_COMMENT = re.compile(r"(^|\s)#([^\!\[])[^\n]*", re.M)  # keep shebang and #! or array indices

def strip_js_comments(src: str)->str:
    src = JS_BLOCK.sub("", src)
    # avoid stripping URLs (http://)
    return JS_LINE.sub(lambda m: "" if "://" not in m.group(0) else m.group(0), src)

def strip_shell_comments(src:str)->str:
    return SHELL_COMMENT.sub(r"\1", src)

def strip_yaml_comments(src:str)->str:
    # remove lines where first non-space is '#'
    out=[]
    for line in src.splitlines(True):
        s=line.lstrip()
        if s.startswith("#"): continue
        out.append(line)
    return "".join(out)

def strip_generic(path: pathlib.Path, src: str)->str:
    suf = path.suffix
    if suf == ".py":
        return strip_python_comments(src)
    if suf in {".js",".ts",".tsx",".jsx"}:
        return strip_js_comments(src)
    if suf in {".sh",".bash",".zsh"}:
        return strip_shell_comments(src)
    if suf in {".yaml",".yml"}:
        return strip_yaml_comments(src)
    # toml: naive – drop full-line comments
    return "\n".join([ln for ln in src.splitlines() if not ln.strip().startswith("#")])

def get_head_file(path: pathlib.Path)->str:
    rel = path.relative_to(ROOT).as_posix()
    try:
        return subprocess.check_output(["git","show",f"HEAD:{rel}"], cwd=ROOT).decode()
    except subprocess.CalledProcessError:
        return ""

def verify_python_ast_equal(before: str, after: str)->bool:
    try:
        return ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
    except SyntaxError:
        # if it was invalid before or after, fail safe
        return False

def main():
    modified = [ROOT/p for p in git_ls_modified()]
    modified = [p for p in modified if should_check(p)]
    if not modified:
        print("No relevant modified files detected.")
        return 0

    bad = []
    for p in modified:
        before = get_head_file(p)
        after = p.read_text(encoding="utf-8", errors="ignore")

        if p.suffix == ".py":
            if not verify_python_ast_equal(before, after):
                bad.append((p, "python-ast-mismatch"))
                continue
        # Heuristic for others: compare comment-stripped content
        b_stripped = strip_generic(p, before)
        a_stripped = strip_generic(p, after)
        if b_stripped != a_stripped:
            bad.append((p, "non-comment-change"))
    if bad:
        for p, why in bad:
            print(f"[FAIL] {p}: {why}")
        return 2
    print("OK: Only comments changed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())