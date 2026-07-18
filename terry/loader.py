"""Load strategy classes from project files or source strings.

Jesse-authored strategy files are accepted unchanged: their static ``jesse`` imports
are translated to Terry modules before compilation without installing a conflicting
top-level ``jesse`` package into the interpreter.
"""
import ast
import os
import sys
import types

from .strategy import Strategy
from .exceptions import InvalidRoutes, InvalidStrategy


def strategy_path(strategies_dir, name):
    return os.path.join(strategies_dir, name, "__init__.py")


def strategy_exists(strategies_dir, name):
    return os.path.exists(strategy_path(strategies_dir, name))


def load_strategy_class(name, strategies_dir):
    """Import strategies/<name>/__init__.py and return its Strategy subclass named `name`."""
    path = strategy_path(strategies_dir, name)
    if not os.path.exists(path):
        raise InvalidRoutes(f'A strategy with the name of "{name}" could not be found.')
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    return _load_compiled_strategy(name, source, path, f"terry_strategies_{name}")


def load_strategy_from_source(name, source):
    """Compile a strategy from a source string (used by tests / significance minimal strategies)."""
    return _load_compiled_strategy(
        name, source, f"<strategy:{name}>", f"terry_inline_{name}")


class _JesseImportTranslator(ast.NodeTransformer):
    """Translate static Jesse strategy imports to Terry's compatible modules."""

    def visit_ImportFrom(self, node):
        if node.module == "jesse" or (node.module or "").startswith("jesse."):
            node.module = "terry" + node.module[len("jesse"):]
        return node

    def visit_Import(self, node):
        translated = []
        for alias in node.names:
            if alias.name == "jesse":
                translated.append(ast.alias(name="terry", asname=alias.asname or "jesse"))
            elif alias.name.startswith("jesse."):
                translated.append(ast.alias(
                    name="terry" + alias.name[len("jesse"):],
                    asname=alias.asname,
                ))
            else:
                translated.append(alias)
        node.names = translated
        return node


def _load_compiled_strategy(name, source, filename, module_name):
    try:
        tree = ast.parse(source, filename=filename)
        tree = _JesseImportTranslator().visit(tree)
        ast.fix_missing_locations(tree)
    except SyntaxError:
        raise

    module = types.ModuleType(module_name)
    module.__file__ = filename
    # Supports the uncommon ``import jesse.indicators`` form after its import target
    # is translated to ``terry.indicators``; most strategies use ``as ta`` instead.
    module.__dict__["jesse"] = sys.modules.get("terry")
    sys.modules[module_name] = module
    exec(compile(tree, filename, "exec"), module.__dict__)
    cls = getattr(module, name, None)
    if cls is None:
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                cls = obj
                break
    if cls is None or not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise InvalidStrategy(
            f'Strategy "{name}" must define a class named "{name}" that subclasses Strategy.'
        )
    return cls
