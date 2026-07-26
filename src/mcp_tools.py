"""The single tool contract used both locally and by the real MCP server."""

TOOLS_SCHEMA = [
    {"name": "get_sensor_data", "description": "Read zone temperature, outdoor temperature, occupancy, PMV, setpoints and energy.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "propose_setpoints", "description": "Propose HVAC setpoints. Every value is safety-validated before application.", "input_schema": {"type": "object", "properties": {"cooling_setpoint": {"type": "number", "description": "Celsius, 22-28"}, "heating_setpoint": {"type": "number", "description": "Celsius, 16-21.5"}, "fan_flow_fraction": {"type": "number", "description": "0.3-1.0"}, "rationale": {"type": "string"}}, "required": ["cooling_setpoint", "heating_setpoint", "fan_flow_fraction"]}},
    {"name": "get_error_log", "description": "Fetch a deduplicated, length-bounded EnergyPlus error summary for diagnosis.", "input_schema": {"type": "object", "properties": {}}},
]


class MCPToolServer:
    """Tool implementation shared by the orchestrator and stdio MCP server."""
    def __init__(self, ep_session):
        self.ep = ep_session
        self.call_count = 0

    def dispatch(self, tool_name, tool_input=None):
        self.call_count += 1
        tool_input = tool_input or {}
        if tool_name == "get_sensor_data":
            return self.ep.get_state()
        if tool_name == "propose_setpoints":
            result = self.ep.apply_setpoints(**{k: tool_input.get(k) for k in ("cooling_setpoint", "heating_setpoint", "fan_flow_fraction")})
            result["rationale"] = str(tool_input.get("rationale", ""))[:500]
            return result
        if tool_name == "get_error_log":
            return {"log_tail": self.ep.get_error_log_tail()}
        raise ValueError(f"Unknown MCP tool: {tool_name}")
