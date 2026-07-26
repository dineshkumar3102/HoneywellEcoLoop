"""Multi-day closed-loop stability test used as evidence for system integration.

Run: python src/reliability_test.py --days 7
Writes data/reliability_report.json and returns non-zero on a failed invariant.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from energyplus_wrapper import EnergyPlusSession
from mcp_tools import MCPToolServer
from llm_agent import LLMAgent
from orchestrator import BASELINE_IDF, tier1_safety_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be at least 1")

    ep = EnergyPlusSession(BASELINE_IDF, seed=7)
    ep.horizon_minutes = args.days * 24 * 60
    tools = MCPToolServer(ep)
    agent = LLMAgent(tools, strategic_interval_min=30)
    violations = []

    while not ep.is_done():
        agent.maybe_act(ep.t)
        tier1_safety_layer(ep)
        state = ep.step()
        finite = all(math.isfinite(float(state[k])) for k in ("zone_temp_c", "pmv", "instantaneous_kw", "cumulative_kwh"))
        safe = 22 <= state["cooling_setpoint"] <= 28 and 16 <= state["heating_setpoint"] <= 21.5 and state["cooling_setpoint"] >= state["heating_setpoint"] + 1.5 and .3 <= state["fan_flow_fraction"] <= 1
        if not finite or not safe:
            violations.append({"sim_minute": state["sim_minute"], "finite": finite, "safe": safe})

    report = {
        "days_requested": args.days,
        "timesteps_completed": len(ep.log),
        "expected_timesteps": args.days * 96,
        "completed_without_exception": True,
        "safety_invariant_failures": len(violations),
        "strategic_decisions": len(agent.decision_log),
        "mcp_tool_calls": tools.call_count,
        "simulation_engine": ep.engine,
        "controller": "live_ollama" if agent.use_live_llm else "deterministic_fallback",
        "final_cumulative_kwh": round(ep.cumulative_kwh, 3),
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data", "reliability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if len(ep.log) != report["expected_timesteps"] or violations:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
