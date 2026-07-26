# EcoLoop Sync — transparent closed-loop building-control PoC

EcoLoop Sync combines a two-tier HVAC controller, an Ollama-hosted open model, and Model Context Protocol (MCP) tools. The fast tier always enforces hard setpoint/deadband limits; the strategic tier requests setpoints and records every accepted action.

## Runtime truth — important for judges

The included `models/baseline.idf` is a condensed teaching model, so `python src/orchestrator.py` uses the labelled **physics-lite surrogate**. It is functional and reproducible, but it is **not an EnergyPlus result**. The generated dashboard must therefore be described as *surrogate PoC output*, never as an EnergyPlus benchmark.

This revision removes ambiguous claims and adds three live integration paths:

| Capability | Run mode | Evidence |
|---|---|---|
| Open-source LLM tool call | Ollama, model installed locally | `llm_live_endpoint_used: true`, `controller: live_ollama` in `data/decision_log.json` |
| Actual MCP transport | `python src/mcp_server.py` | MCP Inspector / any stdio MCP client can list and call three tools |
| Actual EnergyPlus executable | `python src/orchestrator.py --verify-energyplus --idf <complete.idf>` | `data/energyplus_validation/eplusout.*` and `simulation_engine: EnergyPlus` output |

If a required runtime is missing, strict mode fails with an error; it never quietly claims that a substitute is real.

## Quick demo (surrogate)

```powershell
python -m pip install -r requirements.txt
python src/orchestrator.py
```

This creates `data/run_log.csv`, `data/decision_log.json`, `data/summary.json`, and `models/optimized_run1.idf`. Check the `simulation_engine` and `llm_status` fields before using the results in a presentation.

## Real local LLM: Ollama

Install Ollama, then download a tool-capable local model. On Windows PowerShell:

```powershell
ollama pull llama3.2:3b
$env:LLM_MODEL = "llama3.2:3b"
$env:REQUIRE_LIVE_LLM = "1"
python src/orchestrator.py
```

`REQUIRE_LIVE_LLM=1` deliberately stops the run unless the endpoint and requested model are reachable. This gives a clean, demonstrable proof of genuine tool calling. The action is still bounds-validated server-side.

## Real MCP server (stdio)

```powershell
python src/mcp_server.py
```

Configure an MCP client with command `python` and args `["src/mcp_server.py"]`. It exposes `get_sensor_data`, `propose_setpoints`, and `get_error_log`. Do not print logs to stdout: stdio carries MCP JSON-RPC messages.

## Actual EnergyPlus verification

Install EnergyPlus and prepare a **complete, valid** IDF plus EPW weather file. The supplied compact IDF is not complete enough for the engine. Then:

```powershell
$env:EPLUS_BINARY = "C:\EnergyPlusV25-1-0\energyplus.exe"
$env:EPLUS_WEATHER = "C:\path\to\weather.epw"
python src/orchestrator.py --verify-energyplus --idf C:\path\to\complete_model.idf
```

The command executes the installed `energyplus.exe` and preserves its `eplusout.err` / `eplusout.csv` files in `data/energyplus_validation`. A non-zero engine exit is propagated as a failure. For live per-timestep EnergyPlus write-back, use a complete model with a PythonPlugin or EMS schedule actuator; this repository does not falsely claim that a batch IDF rewrite is live control.

## Safety and resilience

- Bounds and a 1.5°C cooling/heating deadband are enforced below the LLM.
- Tier 1 reacts every 15 simulated minutes while the LLM acts only at strategic intervals.
- Diagnostics are bounded and logged; LLM errors fall back safely and are explicitly marked.
- `ECOLOOP_ENGINE=real` fails without the EnergyPlus binary and EPW instead of using the surrogate.

## Presentation wording

Say: “The delivered demo validates the controller against a labelled RC surrogate. The codebase has a tested real-executable EnergyPlus verification path, real MCP stdio server, and strict live-Ollama mode. Live EnergyPlus callback control is the final integration step once a complete building IDF and weather file are selected.”
