"""A genuine MCP stdio server.

Run: python src/mcp_server.py
Connect with any MCP client using command ``python`` and args
``["src/mcp_server.py"]``.  This is intentionally a separate process: stdio
is reserved for MCP JSON-RPC and must never contain application logging.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from energyplus_wrapper import EnergyPlusSession
from mcp_tools import MCPToolServer

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP support first: pip install -r requirements.txt") from exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
session = EnergyPlusSession(os.path.join(ROOT, "models", "baseline.idf"))
tools = MCPToolServer(session)
mcp = FastMCP("EcoLoop-Sync")

@mcp.tool()
def get_sensor_data() -> dict:
    """Read current live building telemetry."""
    return tools.dispatch("get_sensor_data")

@mcp.tool()
def propose_setpoints(cooling_setpoint: float, heating_setpoint: float, fan_flow_fraction: float, rationale: str = "") -> dict:
    """Submit an HVAC action; server-side safety validation is mandatory."""
    return tools.dispatch("propose_setpoints", locals())

@mcp.tool()
def get_error_log() -> dict:
    """Read compact simulation diagnostics."""
    return tools.dispatch("get_error_log")

if __name__ == "__main__":
    mcp.run(transport="stdio")
