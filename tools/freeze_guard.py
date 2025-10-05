#!/usr/bin/env python3
"""
Freeze guard for demo-freeze-v1

Enforces that, after the freeze, only the following changes are allowed:
- config.yaml policy values only (weights, thresholds, asset_speed_mps, target, c2.*)
- Non-code docs text files (*.md, *.txt, *.rst)
- Optional: comment-only edits in Python files (whitespace or lines starting with '#')

Usage examples:
  python tools/freeze_guard.py --staged
  python tools/freeze_guard.py --base-ref demo-freeze-v1.0

Exit codes:
  0 = OK
  1 = Violations detected
  2 = Invalid invocation or internal error
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    print("freeze_guard: missing dependency pyyaml; please install requirements.txt", file=sys.stderr)
    sys.exit(2)


ALLOWED_DOC_EXT = {".md", ".txt", ".rst"}

DISALLOWED_PATHS = [
    re.compile(r"^tests/"),
    re.compile(r"^out/"),
    re.compile(r"^dev/RECOMM_SCHEMA\.json$"),
]

PY_CODE_PATTERN = re.compile(r"^src/.*\.py$")


def run(cmd: Sequence[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.communicate()[0]
    return p.returncode, out


def git_diff_name_only(base: Optional[str], staged: bool) -> List[str]:
    if staged:
        code, out = run(["git", "diff", "--cached", "--name-only"])
    elif base:
        code, out = run(["git", "diff", f"{base}...HEAD", "--name-only"])
        if code != 0:
            # fallback to range (base..HEAD)
            code, out = run(["git", "diff", f"{base}..HEAD", "--name-only"])
    else:
        code, out = run(["git", "diff", "--name-only"])
    if code != 0:
        print(out)
        raise RuntimeError("git diff failed")
    return [l.strip() for l in out.splitlines() if l.strip()]


def git_file_content(ref: str, path: str) -> Optional[str]:
    code, out = run(["git", "show", f"{ref}:{path}"])
    if code != 0:
        return None
    return out


def load_yaml_from(ref: Optional[str], path: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """Return (base, cur) YAML dicts. base may be None if not found."""
    cur = None
    try:
        cur = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"freeze_guard: failed to load {path}: {e}", file=sys.stderr)
        return None, None
    base = None
    if ref:
        txt = git_file_content(ref, str(path))
        if txt is not None:
            try:
                base = yaml.safe_load(txt)
            except Exception as e:
                print(f"freeze_guard: failed to load base {path} from {ref}: {e}", file=sys.stderr)
                return None, None
    return base, cur


def is_docs_file(p: Path) -> bool:
    return p.suffix.lower() in ALLOWED_DOC_EXT


def is_disallowed_path(path: str) -> Optional[str]:
    for rx in DISALLOWED_PATHS:
        if rx.match(path):
            return f"path blocked by policy: {rx.pattern}"
    return None


def allowed_config_change(base: Optional[dict], cur: Optional[dict]) -> Tuple[bool, List[str]]:
    """Validate that only allowed config keys changed.

    Allowed subtrees/keys:
      - asset_speed_mps
      - weights.* (any key under weights)
      - thresholds.IMMEDIATE*, thresholds.SUSPECT* (to permit both _score and legacy names)
      - target.*
      - c2.*
    Everything else must be identical between base and cur.
    If base is None (first freeze commit), accept.
    """
    if base is None or cur is None:
        return True, []

    violations: List[str] = []

    def dict_get(d: dict, *keys, default=None):
        x = d
        for k in keys:
            if not isinstance(x, dict):
                return default
            if k not in x:
                return default
            x = x[k]
        return x

    # Helper to compare two values and record if changed
    def changed(path: str, a, b) -> bool:
        return a != b

    # Compare top-level keys conservatively
    allowed_top = {"asset_speed_mps", "weights", "thresholds", "target", "c2"}

    # 1) Any unexpected add/remove at top-level?
    a_keys = set(base.keys()) if isinstance(base, dict) else set()
    b_keys = set(cur.keys()) if isinstance(cur, dict) else set()
    extra = (b_keys - a_keys) | (a_keys - b_keys)
    if extra:
        # If new top-level keys are added that are not allowed, flag
        for k in extra:
            if k not in allowed_top:
                violations.append(f"config.yaml: top-level key change not allowed: {k}")

    # 2) asset_speed_mps
    if changed("asset_speed_mps", base.get("asset_speed_mps"), cur.get("asset_speed_mps")):
        pass  # allowed

    # 3) weights subtree: allow any weights.* key to change
    # but do not allow adding/removing 'weights' entirely
    bw = dict_get(base, "weights", default={}) or {}
    cw = dict_get(cur, "weights", default={}) or {}
    # Nothing to do; all keys under weights are allowed to differ

    # 4) thresholds: only keys starting with IMMEDIATE or SUSPECT may differ
    bt = dict_get(base, "thresholds", default={}) or {}
    ct = dict_get(cur, "thresholds", default={}) or {}
    all_thr_keys = set(bt.keys()) | set(ct.keys())
    for k in all_thr_keys:
        if not (k.startswith("IMMEDIATE") or k.startswith("SUSPECT")):
            if changed(f"thresholds.{k}", bt.get(k), ct.get(k)):
                violations.append(f"config.yaml: thresholds.{k} change not allowed")

    # 5) target subtree: allow any target.* key to change
    # 6) c2 subtree: allow any c2.* key to change
    # No explicit checks needed; we will block changes outside allowed subtrees below

    # 7) Block changes outside allowed subtrees/keys by comparing snapshots
    # Remove allowed subtrees from deep copies and compare remainder
    import copy
    ba = copy.deepcopy(base)
    ca = copy.deepcopy(cur)
    for k in ["asset_speed_mps", "weights", "thresholds", "target", "c2"]:
        if isinstance(ba, dict) and k in ba:
            if k == "thresholds":
                # Keep only disallowed keys in thresholds for comparison
                bt2 = dict_get(ba, "thresholds", default={}) or {}
                bt_keep = {kk: vv for kk, vv in bt2.items() if not (kk.startswith("IMMEDIATE") or kk.startswith("SUSPECT"))}
                ba["thresholds"] = bt_keep
            else:
                ba.pop(k, None)
        if isinstance(ca, dict) and k in ca:
            if k == "thresholds":
                ct2 = dict_get(ca, "thresholds", default={}) or {}
                ct_keep = {kk: vv for kk, vv in ct2.items() if not (kk.startswith("IMMEDIATE") or kk.startswith("SUSPECT"))}
                ca["thresholds"] = ct_keep
            else:
                ca.pop(k, None)

    if ba != ca:
        violations.append("config.yaml: changes outside allowed keys detected")

    return len(violations) == 0, violations


def py_comment_only_changes(base_ref: str, path: str, staged: bool) -> Tuple[bool, List[str]]:
    """Return True if all changed lines are comments/blank in a .py file."""
    if staged:
        cmd = ["git", "diff", "--cached", "-U0", "--", path]
    else:
        cmd = ["git", "diff", "-U0", f"{base_ref}...HEAD", "--", path]
    code, out = run(cmd)
    if code != 0:
        return False, [f"git diff failed for {path}"]
    violations: List[str] = []
    for line in out.splitlines():
        if not line:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            content = line[1:]
            stripped = content.strip()
            if stripped == "":
                continue
            if stripped.startswith("#"):
                continue
            violations.append(f"{path}: code change beyond comments detected: '{stripped[:60]}'")
    return len(violations) == 0, violations


@dataclass
class GuardResult:
    ok: bool
    messages: List[str]


def guard(base_ref: Optional[str], staged: bool) -> GuardResult:
    messages: List[str] = []

    # Resolve base_ref fallback if not provided
    if not staged and not base_ref:
        # Try the tag first, then branch name
        code, out = run(["git", "rev-parse", "--verify", "demo-freeze-v1.0^{tag}"])
        if code == 0:
            base_ref = "demo-freeze-v1.0"
        else:
            code, out = run(["git", "rev-parse", "--verify", "origin/demo-freeze-v1"])
            if code == 0:
                base_ref = "origin/demo-freeze-v1"

    try:
        changed_files = git_diff_name_only(base_ref, staged)
    except Exception as e:
        return GuardResult(False, [f"freeze_guard: failed to compute diff: {e}"])

    if not changed_files:
        return GuardResult(True, ["no changes detected"])

    violations: List[str] = []

    # Evaluate each changed file
    for f in changed_files:
        p = Path(f)

        # Block certain paths immediately
        block_reason = is_disallowed_path(f)
        if block_reason:
            violations.append(f"{f}: {block_reason}")
            continue

        if f == "config.yaml":
            base_cfg, cur_cfg = load_yaml_from(base_ref, p)
            ok, errs = allowed_config_change(base_cfg, cur_cfg)
            if not ok:
                violations.extend(errs)
            continue

        # Allow docs ext anywhere
        if is_docs_file(p):
            continue

        # For Python code under src/, allow only comment/blank changes
        if PY_CODE_PATTERN.match(f):
            if base_ref is None and not staged:
                # If no base ref and not staged diff, we cannot validate safely
                violations.append(f"{f}: cannot validate without base ref; provide --base-ref or use --staged")
                continue
            ok, errs = py_comment_only_changes(base_ref or "HEAD", f, staged)
            if not ok:
                violations.extend(errs)
            continue

        # Otherwise, block changes by default
        violations.append(f"{f}: changes not allowed under freeze policy")

    ok = len(violations) == 0
    if not ok:
        messages.append("Freeze guard violations:\n  - " + "\n  - ".join(violations))
    else:
        messages.append("freeze guard: OK")
    return GuardResult(ok, messages)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Enforce demo-freeze-v1 change policy")
    ap.add_argument("--base-ref", default=None, help="Git ref to diff against (e.g., demo-freeze-v1.0 or origin/demo-freeze-v1)")
    ap.add_argument("--staged", action="store_true", help="Validate staged changes instead of a ref range")
    args = ap.parse_args(argv)

    res = guard(args.base_ref, args.staged)
    for m in res.messages:
        print(m)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())

