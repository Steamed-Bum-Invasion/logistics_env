# AGENTS.md - hackathon_env

## Commands

```bash
uv sync                          # install deps
uvicorn server.app:app --reload  # run dev server on :8000
uv run --project . server        # run via pyproject entry point
```

## Architecture

Logistics chain management environment using **MCP (FastMCP) tool pattern**.

```
hackathon_env/
├── models.py              # LogiChainObservation, LogiChainToolObservation, LogiChainState
├── client.py              # HackathonEnv WebSocket client
├── server/
│   ├── app.py             # FastAPI app (uses base Action, not custom)
│   ├── hackathon_env_environment.py  # LogiChainEnvironment + _LogiChainMCPDelegate
│   └── network_graph.py   # NetworkX-based logistics network (12 nodes, 18 edges)
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
| `advance_shift` | Advance time, process deliveries, spawn new orders |
| `query_driver` | Get driver status/location |
| `query_order` | Get order status/details |
| `query_network` | Get network topology + traffic |

## Key Mechanics

- **Action budget**: 3 actions per shift, decrements on mutation tools, resets on `advance_shift()`
- **New orders**: Spawn on every `advance_shift()` call (1-3 random orders)
- **Time**: Only advances via `advance_shift()` tool (not implicit per step)
- **Done**: When `time_step >= max_steps` (50) or all orders delivered/failed
- **Rewards**: Terminal on `advance_shift()` (delivery success/failure), shaping rewards on other tools

## NetworkGraph

- 12 nodes: 1 hub, 2 warehouses, 5 zones, 4 customer areas
- 18 bidirectional roads with distance, capacity, road_type
- Traffic multipliers affect shortest path costs (NetworkX Dijkstra)
- `update_traffic(multiplier=1.3)` for traffic spike events

## Observation Types

- `LogiChainObservation` — returned by `reset()`, includes dashboard + available_tools
- `LogiChainToolObservation` — returned by `step()` for tool calls, includes tool_result + reward
- Both include `available_tools` and `remaining_actions` fields (always visible to agent)
