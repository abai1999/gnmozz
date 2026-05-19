#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFINER_PATH = ROOT / "prismatic" / "robot" / "stage_aware_refiner.py"


ALLOWED_SELF_PREFIXES = (
    "_b1_group_shadow",
    "_last_b1_group_shadow",
    "_b1_apply_gate",
    "_last_b1_apply_gate",
)

FORBIDDEN_SELF_NAMES = {
    "_last_scorer_candidate_index",
    "_last_scorer_group_index",
    "_held_alignment_candidate_idx",
    "_held_alignment_candidate_group",
    "_last_selected_step_scale",
    "_runtime_handoff_ready",
}

FORBIDDEN_LOCAL_TARGETS = {
    "group_logits",
    "scores",
    "candidate_scores",
    "candidate_mask",
    "group_mask",
    "masked_scores",
    "pred_idx",
    "selected_idx",
    "action",
}


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        prefix = ""
        if isinstance(target.value, ast.Name):
            prefix = target.value.id + "."
        return [prefix + target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    if isinstance(target, ast.Subscript):
        return _target_names(target.value)
    return []


def _is_shadow_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_b1_group_selector_shadow"
    )


def main() -> int:
    source = REFINER_PATH.read_text()
    tree = ast.parse(source, filename=str(REFINER_PATH))
    errors: list[str] = []
    warnings: list[str] = []

    shadow_fn: ast.FunctionDef | None = None
    call_nodes: list[ast.AST] = []
    assigned_call_nodes: list[ast.AST] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_b1_group_selector_shadow":
            shadow_fn = node
        if _is_shadow_call(node):
            call_nodes.append(node)

    if shadow_fn is None:
        errors.append("_run_b1_group_selector_shadow is missing")
    else:
        for node in ast.walk(shadow_fn):
            if isinstance(node, ast.Return) and node.value is not None:
                errors.append(f"shadow function returns a value at line {node.lineno}")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        targets.extend(_target_names(target))
                else:
                    targets.extend(_target_names(node.target))
                for name in targets:
                    if name.startswith("self."):
                        attr = name.split(".", 1)[1]
                        if attr in FORBIDDEN_SELF_NAMES:
                            errors.append(f"shadow writes forbidden self.{attr} at line {node.lineno}")
                        elif not attr.startswith(ALLOWED_SELF_PREFIXES):
                            errors.append(f"shadow writes non-shadow self.{attr} at line {node.lineno}")
                    elif name in FORBIDDEN_LOCAL_TARGETS:
                        errors.append(f"shadow writes suspicious main-control local {name} at line {node.lineno}")

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if _is_shadow_call(child):
                if not isinstance(parent, ast.Expr):
                    assigned_call_nodes.append(child)

    if not call_nodes:
        errors.append("_run_b1_group_selector_shadow is never called")
    if assigned_call_nodes:
        for node in assigned_call_nodes:
            errors.append(f"shadow call result is consumed at line {getattr(node, 'lineno', '?')}")

    if len(call_nodes) > 1:
        warnings.append(f"shadow function has {len(call_nodes)} call sites; review all of them")

    report = {
        "ok": not errors,
        "file": str(REFINER_PATH.relative_to(ROOT)),
        "shadow_call_count": len(call_nodes),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
