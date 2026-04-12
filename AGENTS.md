# AGENTS.md - logistics_env

## Commands

```bash
uv sync                          # install deps
uvicorn server.app:app --reload  # run dev server on :8000
uv run --project . server        # run via pyproject entry point
uv run python inference.py       # run inference with LLM
```

## Inference Setup

For local development, create `.env` from `.env.example`:
```
OPENROUTER_API_KEY=your_key_here
```

For submission, the system injects `API_KEY` and `API_BASE_URL` at runtime. Defaults in `inference.py`:
- `API_BASE_URL` = `https://openrouter.ai/api/v1`
- `MODEL_NAME` = `qwen/qwen-2.5-72b-instruct`

## Architecture

MCP (FastMCP) tool pattern with two layers:
- `_LogiChainMCPDelegate` extends `MCPEnvironment` — holds FastMCP server, delegates tool dispatch
- `LogiChainEnvironment` extends base `Environment` — controls rewards, state, episode lifecycle

**Key files:**
```
├── models.py              # LogiChainAction, LogiChainObservation, LogiChainToolObservation
├── client.py              # LogisticsEnv WebSocket client (extends EnvClient)
├── inference.py           # LLM inference script
└── server/
    ├── app.py             # FastAPI app (uses base Action, not custom)
    ├── logistics_environment.py  # Core environment + MCP delegate
    ├── tools.py           # 7 tool implementations
    ├── rewards.py         # Reward computation
    ├── network_graph.py   # NetworkX-based network
    ├── task_config.yaml   # Task definitions
    └── reward_config.yaml # Reward tuning
```

## Tools

| Tool | Purpose |
|------|---------|
| `assign_order` | Assign pending order to driver |
| `reroute_driver` | Reroute driver to order dropoff |
| `delay_order` | Extend deadline +3 steps |
| `escalate_order` | Extend deadline +5 steps |
| `query_driver` | Get driver status/location |
| `query_order` | Get order status/details |
| `query_network` | Get network topology + traffic |

## Critical Mechanics

- **Rerouting required**: After pickup, must call `reroute_driver` or driver stays idle
- **Driver reassignment**: Assigning a busy driver abandons their current orders (reverts to "pending") + `-0.1` penalty
- **Time advance**: Every `step()` call advances time by 1 tick
- **Deadlines**: Orders fail immediately when `time_step > deadline`
- **No-progress**: After 10 steps without delivery/active orders, `-0.05` per step

## Reward Structure

See `server/reward_config.yaml` for tunable values. Defaults:
- On-time delivery: `+1.0`
- Late delivery: `+0.5`
- Failed: `-0.5`
- Reassignment penalty: `-0.1`

## Tasks

Defined in `server/task_config.yaml`. All use `variation_mode: fixed` (same network each reset).

| Task | Time Limit | Key Config |
|------|------------|------------|
| `speed_run` | 20 steps | Small network, 5 drivers |
| `quick_delivery` | 30 steps | Default network |
| `on_time_efficiency` | All resolved | Larger network, 8 drivers |
| `deadline_priority` | All resolved | Complex network, tight deadlines |
| `throughput_master` | All resolved | Large network, 10 drivers |

## Network Configuration

Networks can be:
1. **Default** (no config) — 12 nodes, 18 edges
2. **Procedural** — specify `num_hubs`, `num_warehouses`, etc. in task config
3. **Explicit** — `config_file: networks/custom.yaml`

Network files in `server/networks/`: `default.yaml`, `small_city.yaml`, `complex.yaml`
