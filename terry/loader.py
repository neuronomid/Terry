"""Load strategy classes from the project's strategies/ directory or from source strings."""
import importlib.util
import os
import sys

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

    module_name = f"terry_strategies_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, name, None)
    if cls is None:
        # fall back: first Strategy subclass defined in the module
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                cls = obj
                break
    if cls is None or not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise InvalidStrategy(
            f'Strategy file for "{name}" must define a class named "{name}" that subclasses Strategy.'
        )
    return cls


def load_strategy_from_source(name, source):
    """Compile a strategy from a source string (used by tests / significance minimal strategies)."""
    module_name = f"terry_inline_{name}"
    module = type(sys)(module_name)
    sys.modules[module_name] = module
    exec(compile(source, f"<strategy:{name}>", "exec"), module.__dict__)
    cls = getattr(module, name, None)
    if cls is None:
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                cls = obj
                break
    if cls is None:
        raise InvalidStrategy(f'Inline strategy "{name}" must define a class named "{name}".')
    return cls
