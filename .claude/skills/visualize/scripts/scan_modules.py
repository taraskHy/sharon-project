"""Static module inventory for the flow map (stdlib only, no LLM).

Walks one or more Python packages, extracts per-module facts via ast, and
writes a scan.json that the flow-map renderer uses for drill-down detail and
that Claude uses as a completeness checklist while authoring flow-spec.json.

This scan is EVIDENCE, not the architecture: it sees imports and direct call
names, but not dependency injection, subprocess chaining, or runtime config.
Those runtime flows belong in flow-spec.json, verified in source.

Usage:
    python scan_modules.py [package_dir ...] [--root .] [--out flow-out/scan.json]

Defaults to scanning the `autograder` package under the current directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ENTRY_HINT_IMPORTS = {
    "argparse": "argparse",
    "click": "click",
    "typer": "typer",
    "flask": "flask",
    "fastapi": "fastapi",
    "streamlit": "streamlit",
    "gradio": "gradio",
    "http.server": "http.server",
    "subprocess": "subprocess",
}


def module_name_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative(module: str, level: int, current: str) -> str | None:
    """Resolve `from ..x import y` to an absolute dotted module name."""
    base = current.split(".")
    # level 1 = current package: drop the module segment itself
    if level > len(base):
        return None
    prefix = base[: len(base) - level]
    if module:
        prefix = prefix + module.split(".")
    return ".".join(prefix) if prefix else None


class ModuleScanner(ast.NodeVisitor):
    def __init__(self, modname: str, known_prefixes: tuple[str, ...]):
        self.modname = modname
        self.known_prefixes = known_prefixes
        self.imports: set[str] = set()          # internal absolute module names
        self.external_imports: set[str] = set()  # top-level external names
        self.imported_names: dict[str, str] = {}  # local name -> internal module
        self.defs: list[dict] = []
        self.entry_hints: set[str] = set()
        self.calls: list[dict] = []              # {"to": module, "name": fn, "line": n}
        self._depth = 0

    # -- imports --------------------------------------------------------
    def _note_import(self, absname: str, asname: str | None, leaf: str | None):
        if absname is None:
            return
        top = absname.split(".")[0]
        if any(absname == p or absname.startswith(p + ".") for p in self.known_prefixes):
            self.imports.add(absname)
            local = asname or (leaf if leaf else absname.split(".")[0])
            self.imported_names[local] = absname
        else:
            self.external_imports.add(top)
            if top in ENTRY_HINT_IMPORTS:
                self.entry_hints.add(ENTRY_HINT_IMPORTS[top])

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._note_import(alias.name, alias.asname, None)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level:
            base = resolve_relative(node.module or "", node.level, self.modname)
        else:
            base = node.module
        if base is None:
            return
        internal = any(base == p or base.startswith(p + ".") for p in self.known_prefixes)
        for alias in node.names:
            if alias.name == "*":
                self._note_import(base, None, None)
                continue
            child = f"{base}.{alias.name}"
            if internal:
                # `from pkg import module` vs `from pkg.module import name`
                self.imports.add(base)
                self.imported_names[alias.asname or alias.name] = child
            else:
                self._note_import(base, None, None)

    # -- definitions ----------------------------------------------------
    def _def(self, node, kind: str):
        if self._depth == 0:
            end = getattr(node, "end_lineno", node.lineno)
            self.defs.append(
                {"name": node.name, "kind": kind, "line": node.lineno, "loc": end - node.lineno + 1}
            )
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, node):
        self._def(node, "def")

    def visit_AsyncFunctionDef(self, node):
        self._def(node, "async def")

    def visit_ClassDef(self, node):
        self._def(node, "class")

    # -- calls & entry hints --------------------------------------------
    def visit_Call(self, node: ast.Call):
        target = None
        name = None
        f = node.func
        if isinstance(f, ast.Name):
            name = f.id
            target = self.imported_names.get(f.id)
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            base = self.imported_names.get(f.value.id)
            if base:
                target = base
                name = f.attr
        if target:
            # `from pkg import module` binds a module: calls look like module.fn
            self.calls.append({"to": target, "name": name, "line": node.lineno})
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        t = node.test
        if (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "__name__"
        ):
            self.entry_hints.add("__main__")
        self.generic_visit(node)


def scan_package(pkg_dir: Path, root: Path, known_prefixes: tuple[str, ...]) -> dict:
    modules: dict[str, dict] = {}
    for py in sorted(pkg_dir.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "venv"} for part in py.parts):
            continue
        modname = module_name_for(py, root)
        try:
            source = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError as exc:
            print(f"  ! skip {py}: {exc}", file=sys.stderr)
            continue
        scanner = ModuleScanner(modname, known_prefixes)
        scanner.visit(tree)
        modules[modname] = {
            "path": py.relative_to(root).as_posix(),
            "loc": source.count("\n") + 1,
            "defs": sorted(scanner.defs, key=lambda d: -d["loc"])[:25],
            "imports": sorted(scanner.imports - {modname}),
            "external": sorted(scanner.external_imports),
            "entry_hints": sorted(scanner.entry_hints),
            "calls": scanner.calls[:400],
        }
    return modules


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packages", nargs="*", default=["autograder"],
                    help="package directories to scan (relative to --root)")
    ap.add_argument("--root", default=".", help="repo root (module names are relative to it)")
    ap.add_argument("--out", default="flow-out/scan.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pkg_dirs = [root / p for p in (args.packages or ["autograder"])]
    for d in pkg_dirs:
        if not d.exists():
            print(f"error: package dir not found: {d}", file=sys.stderr)
            return 2
    prefixes = tuple(d.name for d in pkg_dirs)

    modules: dict[str, dict] = {}
    for d in pkg_dirs:
        modules.update(scan_package(d, root, prefixes))

    # Module-level call edges, aggregated with counts (drop self-edges).
    edge_counts: dict[tuple[str, str], int] = {}
    edge_names: dict[tuple[str, str], set[str]] = {}
    for mod, info in modules.items():
        for call in info["calls"]:
            to_mod = call["to"]
            if to_mod not in modules:
                # `from pkg.module import name` recorded pkg.module.name
                parent = to_mod.rsplit(".", 1)[0]
                if parent in modules:
                    to_mod = parent
                else:
                    continue
            if to_mod == mod:
                continue
            key = (mod, to_mod)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            if call["name"]:
                edge_names.setdefault(key, set()).add(call["name"])

    call_edges = [
        {"from": a, "to": b, "count": n, "names": sorted(edge_names.get((a, b), set()))[:8]}
        for (a, b), n in sorted(edge_counts.items())
    ]
    import_edges = sorted(
        {(m, i) for m, info in modules.items() for i in info["imports"] if i in modules and i != m}
    )

    for info in modules.values():
        del info["calls"]  # raw per-call rows are only needed for aggregation

    out = {
        "root": root.as_posix(),
        "packages": [d.name for d in pkg_dirs],
        "module_count": len(modules),
        "modules": modules,
        "import_edges": [list(e) for e in import_edges],
        "call_edges": call_edges,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"scanned {len(modules)} modules -> {out_path}")
    entry = [m for m, i in modules.items() if i["entry_hints"]]
    print(f"entry-point hints: {', '.join(sorted(entry)) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
