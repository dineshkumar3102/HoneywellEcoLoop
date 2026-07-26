"""Plain-text console demonstration of the EcoLoop control loop.

Suggested recording command (about 90 seconds):
    python src/live_demo.py --hours 24
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from energyplus_wrapper import EnergyPlusSession
from llm_agent import LLMAgent
from mcp_tools import MCPToolServer
from orchestrator import BASELINE_IDF, tier1_safety_layer


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="EcoLoop Sync console demonstration")
    parser.add_argument("--hours", type=float, default=24.0, help="simulated hours to show")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds to pause after each strategic decision")
    parser.add_argument("--fast", action="store_true", help="use a 0.25-second delay")
    parser.add_argument("--no-refresh-dashboard", action="store_false", dest="refresh_dashboard", help="do not regenerate dashboard data after the demo")
    parser.set_defaults(refresh_dashboard=True)
    args = parser.parse_args()
    if args.hours <= 0 or args.delay < 0:
        raise SystemExit("--hours must be positive and --delay cannot be negative")
    delay = 0.25 if args.fast else args.delay

    section("EcoLoop Sync | Closed-loop HVAC demonstration")
    ep = EnergyPlusSession(BASELINE_IDF, seed=7)
    ep.horizon_minutes = int(args.hours * 60)
    tools = MCPToolServer(ep)
    agent = LLMAgent(tools, strategic_interval_min=30)
    print(f"Simulation mode : {ep.engine}")
    print(f"Controller mode : {'Live Ollama' if agent.use_live_llm else 'Local fallback policy'}")
    print("Control cadence : one strategic decision every 30 simulated minutes")
    print(f"Recording pace  : {delay:.2f} seconds per decision")

    decision_count = 0
    while not ep.is_done():
        decision = agent.maybe_act(ep.t)
        if decision:
            decision_count += 1
            state, request, applied = decision["state"], decision["decision"], decision["applied"]
            section(f"Decision {decision_count:02d} | simulated time {state['hour_of_day']:05.2f} h")
            print("Telemetry")
            print(f"  Zone temperature : {state['zone_temp_c']:.2f} C    Outdoor : {state['outdoor_temp_c']:.2f} C")
            print(f"  Occupancy        : {state['occupancy_fraction']:.2f}       PMV     : {state['pmv']:.2f}")
            print(f"  Current demand   : {state['instantaneous_kw']:.3f} kW   Energy  : {state['cumulative_kwh']:.3f} kWh")
            print("Control request")
            print(f"  Cooling setpoint : {request['cooling_setpoint']:.1f} C    Heating : {request['heating_setpoint']:.1f} C")
            print(f"  Fan flow         : {request['fan_flow_fraction']:.2f}")
            print(f"  Reason           : {request['rationale']}")
            print("Safety validation")
            print(f"  Applied cooling  : {applied['applied_cooling_setpoint']:.1f} C    Applied heating : {applied['applied_heating_setpoint']:.1f} C")
            print(f"  Applied fan flow : {applied['applied_fan_flow_fraction']:.2f}")
            time.sleep(delay)
        tier1_safety_layer(ep)
        ep.step()

    section("Demonstration complete")
    print(f"Strategic decisions : {decision_count}")
    print(f"MCP tool calls      : {tools.call_count}")
    print(f"Total energy        : {ep.cumulative_kwh:.2f} kWh")
    if args.refresh_dashboard:
        print("Refreshing baseline/adaptive dashboard data...")
        completed = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "orchestrator.py")])
        if completed.returncode:
            raise SystemExit(completed.returncode)
        print("Dashboard data refreshed. Open dashboard/dashboard.html and press Ctrl+F5.")
    else:
        print("Dashboard refresh disabled. Run src/orchestrator.py to regenerate the comparison.")


if __name__ == "__main__":
    main()
