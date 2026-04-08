# AGENTS.md - logistics_env

## Commands

```bash
uv sync                          # install deps
uvicorn server.app:app --reload  # run dev server on :8000
uv run --project . server        # run via pyproject entry point
uv run python inference.py       # run inference with LLM
```

## Architecture

Logistics chain management environment using **MCP (FastMCP) tool pattern**.

```
logistics_env/
├── models.py              # LogiChainObservation, LogiChainToolObservation, LogiChainState
├── client.py              # LogisticsEnv WebSocket client
├── inference.py           # LLM inference script
└── server/
    ├── app.py             # FastAPI app (uses base Action, not custom)
    ├── logistics_environment.py  # LogiChainEnvironment + _LogiChainMCPDelegate
    ├── tools.py           # Tool implementations (assign_order, query_*, etc.)
    ├── rewards.py         # Reward computation logic
    ├── grader.py          # Task grading logic
    └── network_graph.py  # NetworkX-based logistics network (12 nodes, 18 edges)
```

## MCP Pattern

- `_LogiChainMCPDelegate` extends `MCPEnvironment` — holds FastMCP server, delegates tool dispatch
- `LogiChainEnvironment` extends base `Environment` — controls rewards, state, episode lifecycle
- `app.py` passes **base `Action`** to `create_app()` (not a custom action class)
- Agent sends `CallToolAction` or `ListToolsAction` via `step()`

## Available Tools

| Tool | Purpose |
|------|---------|
| `assign_order` | Assign pending order to driver |
| `reroute_driver` | Reroute driver to order dropoff |
| `delay_order` | Extend deadline +3 steps |
| `escalate_order` | Extend deadline +5 steps (costs more) |
| `query_driver` | Get driver status/location |
| `query_order` | Get order status/details |
| `query_network` | Get network topology + traffic |

## Key Mechanics

- **Time advance**: Every `step()` call advances time by 1 tick automatically
- **Driver movement**: Drivers move 1 unit toward destination every step
- **Deadlines**: Orders fail immediately when `time_step > deadline` (continuous check)
- **New orders**: 20% chance per step to spawn 1-2 new orders
- **Traffic**: 15% chance per step for traffic spike (1.3x multiplier)
- **Done**: When `time_step >= max_steps` (50) or all orders delivered/failed

## NetworkGraph

- 12 nodes: 1 hub, 2 warehouses, 5 zones, 4 customer areas
- 18 bidirectional roads with distance, capacity, road_type
- Traffic multipliers affect shortest path costs (NetworkX Dijkstra)
- `update_traffic(multiplier=1.3)` for traffic spike events

## Observation Types

- `LogiChainObservation` — returned by `reset()`, includes dashboard + available_tools
- `LogiChainToolObservation` — returned by `step()` for tool calls, includes tool_result + reward
- Both include `available_tools` and `time_step` fields (always visible to agent)

## Reward Structure

| Event | Reward |
|-------|--------|
| Order delivered on time | +1.0 |
| Order delivered late | +0.5 |
| Order failed (deadline exceeded) | -0.5 |

## Tasks

| Task | Time Limit | Objective |
|------|------------|-----------|
| speed_run | 20 steps | Max deliveries quickly |
| quick_delivery | 30 steps | Deliver before pile-up |
| on_time_efficiency | All resolved | Maximize on-time |
| deadline_priority | All resolved | Handle tight deadlines |
| throughput_master | All resolved | High volume + efficiency |
