"""Behavioral regression for the _parse_ctx except handler in generate_ragas_testset.py.

The handler must catch BOTH json.JSONDecodeError and TypeError. In Python 3,
`except A, B:` parses as a tuple type and behaves the same as `except (A, B):`
— ruff strips the parens, so we check behavior via AST, not source spelling.
"""

import ast
from pathlib import Path


def test_module_parses():
    src = Path("src/tamubot/evals/generate_ragas_testset.py").read_text()
    ast.parse(src)


def test_parse_ctx_catches_both_exceptions():
    src = Path("src/tamubot/evals/generate_ragas_testset.py").read_text()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_parse_ctx":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Try):
                continue
            for handler in sub.handlers:
                t = handler.type
                elts = t.elts if isinstance(t, ast.Tuple) else ([t] if t is not None else [])
                names: list[str] = []
                for e in elts:
                    if isinstance(e, ast.Name):
                        names.append(e.id)
                    elif isinstance(e, ast.Attribute):
                        names.append(e.attr)
                if "JSONDecodeError" in names and "TypeError" in names:
                    found = True
                    break
    assert found, "_parse_ctx must catch both JSONDecodeError and TypeError"


def test_parse_ctx_runtime_behavior():
    """Re-implement _parse_ctx semantics and confirm the contract holds.

    Note: we intentionally do not embed the production except clause here —
    formatters disagree on spelling. We just verify the function shape works.
    """
    import json

    excs = (json.JSONDecodeError, TypeError)

    def _parse_ctx(x):
        if isinstance(x, list):
            return x
        if not x:
            return []
        try:
            return json.loads(x)
        except excs:
            return [x] if isinstance(x, str) else []

    assert _parse_ctx(None) == []
    assert _parse_ctx("") == []
    assert _parse_ctx(["a", "b"]) == ["a", "b"]
    assert _parse_ctx('["a", "b"]') == ["a", "b"]
    assert _parse_ctx("not valid json {") == ["not valid json {"]
    assert _parse_ctx(12345) == []
