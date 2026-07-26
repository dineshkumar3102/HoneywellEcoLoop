# System Architecture — AI-Driven Closed-Loop Building Energy Control

## 1. Overview

This PoC closes the loop between a building energy simulation (EnergyPlus) and
an open-source LLM agent, using a standardized tool-calling interface (MCP
pattern) so the LLM never touches simulation internals directly — it only
calls named tools with validated arguments.

```
 ┌────────────────────┐   sensor state    ┌──────────────────┐
 │   EnergyPlus /      │ ───────────────▶  │   MCP Tool Server │
 │   Physics Session   │                   │  (mcp_tools.py)   │
 │ (energyplus_wrapper)│ ◀─────────────────│                   │
 └────────────────────┘  setpoints/actions └─────────┬─────────┘
          ▲                                           │ tool schema
          │ Tier-1 fast safety layer (every timestep) │
          │                                           ▼
          │                                 ┌───────────────────┐
          └─────────────────────────────────│    LLM Agent      │
             forward-injected setpoints     │  (llm_agent.py)    │
             every strategic interval       │  Tier-2 reasoning   │
                                             └───────────────────┘
```

## 2. Tool-Calling Architecture

Three tools are exposed to the LLM via the MCP tool schema
(`src/mcp_tools.py`):

| Tool | Direction | Purpose |
|---|---|---|
| `get_sensor_data` | EnergyPlus → LLM | Zone temp, outdoor temp, occupancy, PMV, current setpoints, power draw |
| `propose_setpoints` | LLM → EnergyPlus | Cooling/heating setpoints + fan flow fraction + rationale |
| `get_error_log` | EnergyPlus → LLM | Compressed tail of simulation warnings/errors |

The LLM is never allowed to write directly into the IDF or simulation state.
Every `propose_setpoints` call passes through a **hard bounds-validation
layer** in `EnergyPlusSession.apply_setpoints()` (clipping to safe HVAC
ranges, enforcing a minimum heating/cooling deadband) before being applied —
this prevents hallucinated or out-of-range setpoints from ever reaching the
building model, regardless of what the LLM outputs.

In a full deployment, these three tool definitions would be served over a
real MCP transport (stdio or HTTP) so any MCP-compatible LLM client could
attach to the same building session. For this PoC, the tool objects are
invoked in-process (`MCPToolServer.dispatch`) to keep the demo runnable with
zero external services — swapping in a real transport is a drop-in change
that does not touch the agent's reasoning logic.

## 3. Prompt Engineering Strategy

The system prompt fixes the agent's objective and hard constraints
explicitly rather than relying on it to infer them:

- **Objective**: minimize kWh while keeping PMV within [-0.5, 0.5].
- **Hard constraints**: never propose setpoints outside 22-28°C cooling /
  16-21.5°C heating (stated in-prompt *and* enforced in code as defense in
  depth).
- **Forced tool use**: `tool_choice: "required"` so every strategic call
  ends in a concrete `propose_setpoints` action rather than free-text
  advice — the loop needs an action, not a discussion.
- **Compact context**: the user turn contains only the current sensor
  snapshot (a small JSON object) plus a compressed error-log tail, not the
  full simulation history. This keeps token usage flat regardless of how
  long the simulation has been running (see §5).
- **Rationale field**: `propose_setpoints` requires a short `rationale`
  string. This costs little and produces an auditable decision trail
  (`data/decision_log.json`) that doubles as documentation for judges and
  as a debugging aid when a decision looks wrong.

## 4. Prompt Latency Management — Two-Tier Control

Calling an LLM at every 15-minute EnergyPlus timestep is both too slow and
unnecessary — building thermal dynamics change on the order of tens of
minutes, not seconds. This PoC uses a **two-tier control split**:

- **Tier 1 — fast deterministic safety layer** (`tier1_safety_layer` in
  `src/orchestrator.py`): runs every single timestep, no LLM call involved.
  If PMV drifts past a hard comfort limit between the agent's strategic
  decisions, it immediately trims fan authority to correct — a pure
  rule-based reflex, sub-millisecond.
- **Tier 2 — strategic LLM reasoning** (`LLMAgent.maybe_act`): invoked on a
  configurable cadence (default every 30 simulated minutes), where it has
  time to actually matter. This is where setpoint strategy, occupancy
  anticipation, and unoccupied-hour setback decisions are made.

This mirrors how real BMS/DDC systems are layered in practice: fast local
control loops for safety, slower supervisory optimization for strategy. It
also means the *number* of LLM calls scales with wall-clock decision cadence,
not simulation resolution — a 24h run at 15-min timesteps only needs 48
strategic LLM calls (verified: `llm_tool_calls_made` in `data/summary.json`),
which is what keeps this practical to run against any hosted or local LLM
without incurring per-timestep latency.

## 5. Handling Lengthy Simulation Logs

EnergyPlus `.err` files can run to thousands of lines over a long simulation
(sizing warnings, convergence notices, environment reports). Feeding that
raw into an LLM context on every call would blow through context limits and
add latency/cost with no benefit. The approach here:

1. **Deduplicate & summarize at the source.** `get_error_log_tail()` returns
   only the most recent, distinct warning/error lines rather than the
   cumulative log.
2. **Structured, compact format.** Errors are surfaced as short lines, not
   raw EnergyPlus verbose text blocks, and capped at a fixed character
   budget (`n_chars`) so context usage per call is constant regardless of
   simulation length.
3. **Self-correcting fault handling.** Common, recognizable error classes
   (sizing margin warnings, convergence-tolerance overshoots) are the kind
   of pattern a fault-correction tool/agent can act on directly (e.g.
   relaxing a convergence tolerance or bumping equipment sizing) without
   forwarding the full trace to the LLM at all — only genuinely novel
   errors need to reach the reasoning layer.
4. **Separation of sensor stream from log stream.** The high-frequency
   numeric sensor data (`get_sensor_data`) and the low-frequency log stream
   (`get_error_log`) are separate tools, so a quiet, error-free run never
   pays any log-context cost.

## 6. Closed-Loop Verification

`src/orchestrator.py` runs the **same weather/occupancy seed** through both
a fixed rule-based baseline schedule and the full AI-driven closed loop, so
the comparison in the dashboard isolates the control strategy as the only
variable. Results for the included run are in `data/summary.json` and
visualized in `dashboard/dashboard.html`.

## 7. Path to Production / Real EnergyPlus

This repo runs on a physics-lite surrogate model
(`EnergyPlusSession` in `src/energyplus_wrapper.py`) because the
demo/judging environment used to author this submission had no EnergyPlus
binary or internet access to install one. The wrapper's public API
(`get_state`, `apply_setpoints`, `step`, `dump_idf`) is intentionally the
*only* surface the agent and orchestrator touch — pointing
`USE_REAL_ENERGYPLUS = True` at a real EnergyPlus install (driven via `eppy`
+ the EMS/BCVTB socket, or the Functional Mock-up Interface) requires no
changes to `mcp_tools.py`, `llm_agent.py`, or `orchestrator.py`.
Similarly, `llm_agent.py` already targets a live OSS LLM endpoint
(Ollama-compatible `/v1/chat/completions`) and only falls back to the
deterministic policy when that endpoint isn't reachable — pointing
`OLLAMA_HOST` at a running Llama 3 / Mistral / Qwen instance switches the
agent to genuine LLM-driven tool calls with no code changes.
