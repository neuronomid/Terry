"""
Terry MCP Server — FastMCP over streamable-http (same transport as Jesse's MCP server).

Run directly:
    python -m terry.mcp.server --port 9021 --project /path/to/project
Or via the CLI:
    terry serve
"""
import argparse
import logging

from mcp.server.fastmcp import FastMCP

from . import config as mcp_config
from ..context import TerryContext, set_context

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("terry.mcp.server")


def build_server(port=mcp_config.MCP_PORT, project_root=None):
    set_context(TerryContext(project_root))
    mcp = FastMCP("Terry MCP Server", host=mcp_config.MCP_HOST, port=port, json_response=True)

    from .resources import register_resources
    from .tools import register_tools
    register_resources(mcp)
    register_tools(mcp)
    return mcp


def run(port=mcp_config.MCP_PORT, project_root=None):
    mcp = build_server(port=port, project_root=project_root)
    url = f"http://localhost:{port}/mcp"
    logger.info("Starting Terry MCP streamable-http server at %s", url)
    print(f"\n  ✓ Terry MCP Server running at {url}\n")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terry MCP Server")
    parser.add_argument("--port", type=int, default=mcp_config.MCP_PORT)
    parser.add_argument("--project", type=str, default=None, help="Project root directory")
    args = parser.parse_args()
    run(port=args.port, project_root=args.project)
