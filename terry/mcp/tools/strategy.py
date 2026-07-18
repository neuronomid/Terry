"""Strategy file tools: create / read / write strategy source in strategies/<name>/__init__.py."""
import os

from ...context import get_context
from ...loader import load_strategy_class


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
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        if os.path.exists(path):
            return {"error": "exists", "message": f'Strategy "{name}" already exists. Use write_strategy to overwrite.'}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        err = _validate(name)
        if err:
            return {"status": "created_with_errors", "name": name, "path": path,
                    "validation_error": err,
                    "message": "File written but it does not import cleanly — fix and call write_strategy."}
        return {"status": "created", "name": name, "path": path}

    @mcp.tool()
    def read_strategy(name: str) -> dict:
        """Read the source of strategies/<name>/__init__.py."""
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        if not os.path.exists(path):
            return {"error": "not_found", "message": f'Strategy "{name}" not found.'}
        with open(path) as f:
            return {"name": name, "path": path, "content": f.read()}

    @mcp.tool()
    def write_strategy(name: str, content: str) -> dict:
        """Overwrite (or create) strategies/<name>/__init__.py with new source."""
        ctx = get_context()
        path = os.path.join(ctx.strategies_dir, name, "__init__.py")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        err = _validate(name)
        if err:
            return {"status": "written_with_errors", "name": name, "path": path,
                    "validation_error": err}
        return {"status": "written", "name": name, "path": path}
