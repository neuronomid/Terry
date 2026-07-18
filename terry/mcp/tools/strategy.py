"""Strategy file tools: create / read / write strategy source in strategies/<name>/__init__.py."""
import os
import re

from ...context import get_context
from ...loader import load_strategy_class


_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")


def _validate_name(name):
    if not _VALID_NAME.fullmatch(name or ""):
        return "Strategy names must start with a letter and contain only letters, numbers, or underscores."
    return None


def _validate(name):
    ctx = get_context()
    try:
        load_strategy_class(name, ctx.strategies_dir)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def register_strategy_tools(mcp):
    @mcp.tool()
    def create_strategy(name: str, content: str) -> dict:
        """Create a new strategy file at strategies/<name>/__init__.py.

        Args:
            name: The strategy class/folder name (e.g. "SmaCross").
            content: Complete Python source defining `class <name>(Strategy)`.
        """
        name_error = _validate_name(name)
        if name_error:
            return {"status": "error", "action": "strategy_creation_failed",
                    "error": "invalid_name", "error_type": "validation_error",
                    "strategy_name": name, "message": name_error}
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        if os.path.exists(path):
            return {"status": "error", "action": "strategy_creation_failed",
                    "error": "exists", "error_type": "strategy_exists",
                    "strategy_name": name,
                    "message": f'Strategy "{name}" already exists. Use write_strategy to overwrite.'}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        err = _validate(name)
        if err:
            return {"status": "error", "action": "strategy_population_failed",
                    "error": "invalid_strategy", "error_type": "write_failed",
                    "name": name, "strategy_name": name, "path": path,
                    "creation_success": True,
                    "validation_error": err,
                    "message": "File written but it does not import cleanly — fix and call write_strategy."}
        return {"status": "success", "action": "strategy_created_and_populated",
                "session_status": "created", "name": name, "strategy_name": name,
                "path": path,
                "message": f"Strategy '{name}' created and populated successfully"}

    @mcp.tool()
    def read_strategy(name: str) -> dict:
        """Read the source of strategies/<name>/__init__.py."""
        name_error = _validate_name(name)
        if name_error:
            return {"status": "error", "action": "strategy_read_failed",
                    "error": "invalid_name", "error_type": "validation_error",
                    "strategy_name": name, "message": name_error}
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        if not os.path.exists(path):
            return {"status": "error", "action": "strategy_read_failed",
                    "error": "not_found", "error_type": "strategy_not_found",
                    "strategy_name": name, "message": f'Strategy "{name}" not found.'}
        with open(path) as f:
            return {"status": "success", "action": "strategy_read",
                    "name": name, "strategy_name": name, "path": path,
                    "content": f.read(),
                    "message": f"Strategy '{name}' content read successfully"}

    @mcp.tool()
    def write_strategy(name: str, content: str) -> dict:
        """Overwrite an existing strategies/<name>/__init__.py with new source."""
        name_error = _validate_name(name)
        if name_error:
            return {"status": "error", "action": "strategy_write_failed",
                    "error": "invalid_name", "error_type": "validation_error",
                    "strategy_name": name, "message": name_error}
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        if not os.path.exists(path):
            return {"status": "error", "action": "strategy_write_failed",
                    "error": "not_found", "error_type": "strategy_not_found",
                    "strategy_name": name, "message": f'Strategy "{name}" not found.'}
        with open(path, "w") as f:
            f.write(content)
        err = _validate(name)
        if err:
            return {"status": "error", "action": "strategy_write_failed",
                    "error": "invalid_strategy", "error_type": "write_failed",
                    "name": name, "strategy_name": name, "path": path,
                    "validation_error": err, "message": err}
        return {"status": "success", "action": "strategy_updated",
                "session_status": "written", "name": name, "strategy_name": name,
                "path": path,
                "message": f"Strategy '{name}' content updated successfully"}
