"""Strategic controller backed by a real Ollama OpenAI-compatible endpoint."""
import json
import os
import urllib.error
import urllib.request

from mcp_tools import TOOLS_SCHEMA

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL_NAME = os.environ.get("LLM_MODEL", "llama3.2:3b")
LIVE_LLM_REQUIRED = os.environ.get("REQUIRE_LIVE_LLM", "0") == "1"


class LLMAgent:
    def __init__(self, tool_server, strategic_interval_min=60):
        self.tool_server = tool_server
        self.strategic_interval_min = strategic_interval_min
        self.use_live_llm, self.llm_status = self._check_llm_reachable()
        if LIVE_LLM_REQUIRED and not self.use_live_llm:
            raise RuntimeError("REQUIRE_LIVE_LLM=1 but Ollama is unavailable: " + self.llm_status)
        self.decision_log = []

    def _check_llm_reachable(self):
        try:
            with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
                models = [m.get("name", "") for m in json.loads(response.read()).get("models", [])]
            if not any(name == MODEL_NAME or name.split(":")[0] == MODEL_NAME.split(":")[0] for name in models):
                return False, f"Ollama reachable but model '{MODEL_NAME}' is not installed"
            return True, f"Ollama live: {MODEL_NAME}"
        except Exception as exc:
            return False, f"Ollama unavailable ({type(exc).__name__})"

    def maybe_act(self, sim_minute):
        if sim_minute % self.strategic_interval_min:
            return None
        state = self.tool_server.dispatch("get_sensor_data")
        errors = self.tool_server.dispatch("get_error_log")
        decision = self._call_live_llm(state, errors) if self.use_live_llm else self._fallback(state)
        applied = self.tool_server.dispatch("propose_setpoints", decision)
        record = {"sim_minute": sim_minute, "controller": "live_ollama" if self.use_live_llm else "deterministic_fallback", "llm_status": self.llm_status, "state": state, "decision": decision, "applied": applied}
        self.decision_log.append(record)
        return record

    def _call_live_llm(self, state, errors):
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are EcoLoop's HVAC strategy agent. Minimize energy while holding occupied PMV in [-0.5,0.5]. Call propose_setpoints exactly once. Bounds: cooling 22-28C; heating 16-21.5C; fan 0.3-1.0. Give a concise rationale."},
                {"role": "user", "content": json.dumps({"telemetry": state, "diagnostics": errors}, separators=(",", ":"))},
            ],
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOLS_SCHEMA],
            "tool_choice": {"type": "function", "function": {"name": "propose_setpoints"}},
            "temperature": 0.1,
        }
        try:
            request = urllib.request.Request(f"{OLLAMA_HOST}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                message = json.loads(response.read())["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if len(calls) != 1 or calls[0]["function"]["name"] != "propose_setpoints":
                raise ValueError("model did not return exactly one propose_setpoints tool call")
            args = json.loads(calls[0]["function"]["arguments"])
            required = {"cooling_setpoint", "heating_setpoint", "fan_flow_fraction"}
            if not required.issubset(args):
                raise ValueError("tool call missing required controls")
            return args
        except Exception as exc:
            # A live outage cannot bypass the physical safety layer; record it plainly.
            self.llm_status = f"Live call failed ({type(exc).__name__}); safe fallback used"
            return self._fallback(state)

    @staticmethod
    def _fallback(state):
        occ, pmv = state["occupancy_fraction"], state["pmv"]
        if occ < 0.2:
            return {"cooling_setpoint": 27.5, "heating_setpoint": 16.5, "fan_flow_fraction": 0.3, "rationale": "Fallback: unoccupied setback."}
        if pmv > 0.42:
            return {"cooling_setpoint": 22.8, "heating_setpoint": 20.5, "fan_flow_fraction": 1.0, "rationale": "Fallback: warm comfort edge."}
        if pmv < -0.42:
            return {"cooling_setpoint": 23.5, "heating_setpoint": 20.0, "fan_flow_fraction": 0.75, "rationale": "Fallback: cold comfort edge."}
        return {"cooling_setpoint": 23.0, "heating_setpoint": 19.5, "fan_flow_fraction": 0.6, "rationale": "Fallback: comfort stable; trim fan."}
