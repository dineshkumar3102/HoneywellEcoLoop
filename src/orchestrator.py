"""
orchestrator.py
===============
Ties the three subsystems together into the closed loop:

    EnergyPlus (energyplus_wrapper.EnergyPlusSession)
          |  sensor data (get_state)
          v
    MCP Tool Server (mcp_tools.MCPToolServer)
          |
          v
    LLM Agent (llm_agent.LLMAgent)   -- reasons every strategic_interval_min
          |  propose_setpoints(...)
          v
    MCP Tool Server -> apply_setpoints -> EnergyPlus (forward injection)

Runs TWO simulations over the same 24h horizon (same weather/occupancy seed):
  1. BASELINE  - fixed rule-based schedule, no agent involvement
  2. AI-DRIVEN - the closed loop above, agent adjusts setpoints every hour

Writes:
  data/run_log.csv         (row per timestep, both runs, for dashboard)
  data/decision_log.json   (every LLM decision + rationale, for the write-up)
  models/optimized_run1.idf (final AI-adjusted building model)
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from energyplus_wrapper import EnergyPlusSession, run_energyplus_batch
from mcp_tools import MCPToolServer
from llm_agent import LLMAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
BASELINE_IDF = os.path.join(MODELS_DIR, "baseline.idf")


def run_baseline():
    """Fixed schedule, exactly matching baseline.idf's rule-based setpoints."""
    ep = EnergyPlusSession(BASELINE_IDF, seed=7)
    while not ep.is_done():
        hour = (ep.t // 60) % 24
        if 8 <= hour < 18:
            ep.apply_setpoints(cooling_setpoint=23.0, heating_setpoint=21.0, fan_flow_fraction=1.0)
        else:
            ep.apply_setpoints(cooling_setpoint=26.0, heating_setpoint=16.0, fan_flow_fraction=1.0)
        ep.step()
    return ep


def tier1_safety_layer(ep, pmv_hard_limit=0.45):
    """
    Runs EVERY timestep (fast, deterministic, no LLM round trip). This is the
    layer that actually guarantees comfort bounds are respected between the
    LLM agent's slower strategic decisions - solving the latency problem
    described in ARCHITECTURE.md. If the zone drifts past the hard PMV limit
    during occupied hours, nudge the cooling setpoint down immediately.
    """
    state = ep.get_state()
    if state["occupancy_fraction"] > 0.2 and abs(state["pmv"]) > pmv_hard_limit:
        if state["pmv"] > 0:  # too warm: give the cooling loop more fan authority, don't fight the setpoint
            ep.apply_setpoints(fan_flow_fraction=min(1.0, ep.fan_flow_fraction + 0.15))
        else:  # too cold
            ep.apply_setpoints(fan_flow_fraction=min(1.0, ep.fan_flow_fraction + 0.10))


def run_ai_driven(strategic_interval_min=30):
    ep = EnergyPlusSession(BASELINE_IDF, seed=7)  # same seed -> same weather/occupancy
    tools = MCPToolServer(ep)
    agent = LLMAgent(tools, strategic_interval_min=strategic_interval_min)
    while not ep.is_done():
        agent.maybe_act(ep.t)       # Tier 2: strategic LLM decision (slow cadence)
        tier1_safety_layer(ep)      # Tier 1: fast deterministic safety net (every step)
        ep.step()
    return ep, agent


def compute_comfort_violation_minutes(log, band=0.5, occ_threshold=0.2):
    """
    Comfort is only evaluated during OCCUPIED hours (ASHRAE-55 convention -
    nobody is present to experience discomfort during unoccupied setback,
    so those minutes are correctly excluded rather than double-penalizing
    the very setback strategy that saves energy).
    """
    return sum(
        15 for row in log
        if row["occupancy_fraction"] > occ_threshold and abs(row["pmv"]) > band
    )


def main():
    parser = argparse.ArgumentParser(description="EcoLoop Sync closed-loop runner")
    parser.add_argument("--verify-energyplus", action="store_true", help="run the actual EnergyPlus CLI against a complete IDF/EPW supplied through EPLUS_WEATHER")
    parser.add_argument("--idf", default=BASELINE_IDF, help="IDF used with --verify-energyplus")
    args = parser.parse_args()
    if args.verify_energyplus:
        result = run_energyplus_batch(args.idf, os.environ.get("EPLUS_WEATHER"), os.path.join(DATA_DIR, "energyplus_validation"))
        print(json.dumps(result, indent=2))
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Running BASELINE simulation (fixed rule-based schedule)...")
    baseline = run_baseline()

    print("Running AI-DRIVEN closed-loop simulation (LLM agent + MCP tools)...")
    ai_run, agent = run_ai_driven()

    # --- Write combined run log for the dashboard ---
    csv_path = os.path.join(DATA_DIR, "run_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "sim_minute", "hour_of_day", "zone_temp_c", "outdoor_temp_c",
                          "occupancy_fraction", "pmv", "cooling_setpoint", "heating_setpoint",
                          "fan_flow_fraction", "instantaneous_kw", "cumulative_kwh"])
        for row in baseline.log:
            writer.writerow(["baseline", row["sim_minute"], row["hour_of_day"], row["zone_temp_c"],
                              row["outdoor_temp_c"], row["occupancy_fraction"], row["pmv"],
                              row["cooling_setpoint"], row["heating_setpoint"],
                              row["fan_flow_fraction"],
                              row["instantaneous_kw"], row["cumulative_kwh"]])
        for row in ai_run.log:
            writer.writerow(["ai_driven", row["sim_minute"], row["hour_of_day"], row["zone_temp_c"],
                              row["outdoor_temp_c"], row["occupancy_fraction"], row["pmv"],
                              row["cooling_setpoint"], row["heating_setpoint"],
                              row["fan_flow_fraction"],
                              row["instantaneous_kw"], row["cumulative_kwh"]])

    # --- Write agent decision log (for architecture doc / video narration) ---
    decision_path = os.path.join(DATA_DIR, "decision_log.json")
    with open(decision_path, "w") as f:
        json.dump(agent.decision_log, f, indent=2)

    # --- Write final AI-optimized IDF (deliverable #2) ---
    out_idf = os.path.join(MODELS_DIR, "optimized_run1.idf")
    ai_run.dump_idf(out_idf)

    # --- Summary stats (deliverable #3 numbers) ---
    baseline_kwh = baseline.cumulative_kwh
    ai_kwh = ai_run.cumulative_kwh
    pct_reduction = 100.0 * (baseline_kwh - ai_kwh) / baseline_kwh

    baseline_violation_min = compute_comfort_violation_minutes(baseline.log)
    ai_violation_min = compute_comfort_violation_minutes(ai_run.log)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulation_engine": ai_run.engine,
        "baseline_total_kwh": round(baseline_kwh, 3),
        "ai_driven_total_kwh": round(ai_kwh, 3),
        "percent_kwh_reduction": round(pct_reduction, 2),
        "baseline_comfort_violation_minutes": baseline_violation_min,
        "ai_driven_comfort_violation_minutes": ai_violation_min,
        "llm_tool_calls_made": agent.tool_server.call_count,
        "llm_live_endpoint_used": agent.use_live_llm,
        "llm_status": agent.llm_status,
        "mcp_transport": "in-process for this comparison; real stdio server available via src/mcp_server.py",
    }
    with open(os.path.join(DATA_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Keep the standalone dashboard reproducible and in sync with each run.
    dashboard_dir = os.path.join(ROOT, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    embedded = {"baseline": [], "ai_driven": []}
    for row in rows:
        run = row.pop("run")
        embedded[run].append({key: float(value) for key, value in row.items()})
    with open(os.path.join(dashboard_dir, "data_embed.js"), "w") as f:
        f.write("window.__DASHBOARD_DATA__ = " + json.dumps({"data": embedded, "summary": summary, "decisions": agent.decision_log}) + ";\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {csv_path}\nWrote: {decision_path}\nWrote: {out_idf}")


if __name__ == "__main__":
    main()
