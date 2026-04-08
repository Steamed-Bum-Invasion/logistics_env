---
title: Logistics Env Environment Server
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - logistics
  - reinforcement-learning
---

# Logistics Env Environment

A logistics chain management environment for reinforcement learning. The agent manages a fleet of drivers delivering orders across a city network with dynamic order spawning, traffic conditions, and deadline constraints.

## Overview

The Logistics Env simulates a real-world delivery operation where:

- **5 drivers** start at the central hub and can be assigned to pending orders
- **Orders** spawn dynamically with pickup/dropoff locations and deadlines
- **Network** consists of 12 nodes (hub, warehouses, zones, customer areas) connected by 18 bidirectional roads
- **Traffic** conditions change randomly, affecting travel times
- **Goal**: Deliver as many orders as possible before their deadlines expire

## Environment Description

### Network Topology

```
        C1 (customer)
         |
        Z1 (zone)
       /   \
      W1---Z5---W2
     /      |     \
    Z4-----HUB----Z2
     \      |      \
      C4   Z3      C2
            |
           C3
```

- **12 nodes**: 1 hub, 2 warehouses, 5 zones, 4 customer areas
- **18 bidirectional roads** with varying distances and capacities
- **Traffic multipliers** affect shortest path costs (Dijkstra's algorithm)

### Simulation Mechanics

| Mechanic | Description |
|----------|-------------|
| **Time advance** | Every `step()` call advances time by 1 tick |
| **Driver movement** | Drivers move 1 unit toward destination per step |
| **Order deadlines** | Orders fail immediately when `time_step > deadline` |
| **New orders** | 20% chance per step to spawn 1-2 new orders |
| **Traffic spikes** | 15% chance per step for 1.3x traffic multiplier |
| **Episode end** | When all orders resolved or max steps reached |

## Action Space

The agent uses `LogiChainAction` with MCP-style tool calls:

```python
class LogiChainAction(Action):
    type: Literal["call_tool", "list_tools"]
    tool_name: str
    arguments: Dict[str, Any]
```

### Available Tools

| Tool | Arguments | Purpose |
|------|-----------|---------|
| `assign_order` | `order_id`, `driver_id` | Assign a pending order to a driver |
| `reroute_driver` | `driver_id` | Reroute driver to their order's dropoff |
| `delay_order` | `order_id` | Extend deadline by 3 steps |
| `escalate_order` | `order_id` | Extend deadline by 5 steps (higher cost) |
| `query_driver` | `driver_id` | Get driver status and location |
| `query_order` | `order_id` | Get order status and details |
| `query_network` | (none) | Get network topology and traffic |

### Example Actions

```python
# Query network state
LogiChainAction(type="call_tool", tool_name="query_network", arguments={})

# Assign order O0 to driver D0
LogiChainAction(type="call_tool", tool_name="assign_order", 
                arguments={"order_id": "O0", "driver_id": "D0"})

# Query driver status
LogiChainAction(type="call_tool", tool_name="query_driver",
                arguments={"driver_id": "D0"})
```

## Observation Space

### LogiChainObservation (from `reset()`)

Returned when the environment is reset:

```python
class LogiChainObservation(Observation):
    dashboard_text: str      # Current state of drivers and orders
    alerts: List[str]        # Active alerts (new orders, traffic)
    available_tools: List[str]  # List of available tool names
    episode_id: str          # Unique episode identifier
    time_step: int           # Current simulation time
```

### LogiChainToolObservation (from `step()`)

Returned after each tool call:

```python
class LogiChainToolObservation(Observation):
    tool_name: str           # Name of the tool that was called
    tool_result: str         # Text result from the tool
    error_msg: Optional[str] # Error message if call failed
    alerts: List[str]        # Active alerts
    available_tools: List[str]
    time_step: int
```

### LogiChainState (from `state` property)

Current environment state:

```python
class LogiChainState(State):
    time_step: int           # Current simulation time
    step_count: int          # Total agent actions taken
    orders_pending: int      # Orders waiting to be assigned
    orders_in_transit: int   # Orders being delivered
    orders_delivered: int    # Successfully delivered orders
    orders_failed: int       # Failed/expired orders
    on_time_deliveries: int  # Orders delivered before deadline
```

## Tasks

The environment includes 5 tasks with different objectives:

| Task | Difficulty | Objective | Time Limit |
|------|------------|-----------|------------|
| `speed_run` | Very Easy | Deliver as many orders as possible quickly | 20 steps |
| `quick_delivery` | Easy | Deliver orders before they pile up | 30 steps |
| `on_time_efficiency` | Medium | Maximize on-time deliveries | All resolved |
| `deadline_priority` | Hard | Handle tight-deadline orders first | All resolved |
| `throughput_master` | Very Hard | High volume + on-time efficiency | All resolved |

## Reward Structure

### Terminal Rewards

| Event | Reward |
|-------|--------|
| Order delivered on time | +1.0 |
| Order delivered late | +0.5 |
| Order failed (deadline exceeded) | -0.5 |

### Shaping Rewards

| Action | Reward |
|--------|--------|
| Query (driver/order/network) | +0.01 |
| Successful assignment | +0.02 |
| Successful reroute | +0.02 |
| Delay order | -0.02 |
| Escalate order | -0.05 |
| Invalid action | -0.05 |

### No-Progress Penalty

- After 10 steps without progress (delivery or assignment), a penalty of -0.05 is applied per step
- Episode ends after 50 consecutive steps without progress

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker
- `uv` package manager

### Local Development

```bash
# Clone the repository
git clone https://github.com/Steamed-Bum-Invasion/logistics_env.git
cd logistics_env

# Install dependencies
uv sync

# Run the server locally
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Build

```bash
# Build the Docker image
docker build -t logistics-env:latest -f server/Dockerfile .
```

### Running Inference

```bash
# Set environment variables
export HF_TOKEN="your-huggingface-token"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

# Run inference
uv run python inference.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `API_BASE_URL` | Yes | LLM API endpoint |
| `LOCAL_IMAGE_NAME` | No | Docker image name (default: `logistics-env:latest`) |

## Usage Examples

### Basic Usage

```python
from logistics_env import LogisticsEnv
from logistics_env.models import LogiChainAction

async def run_episode():
    # Create environment from Docker image
    env = await LogisticsEnv.from_docker_image("logistics-env:latest")
    
    try:
        # Reset environment
        result = await env.reset(task_name="quick_delivery")
        print(f"Dashboard:\n{result.observation.dashboard_text}")
        
        # Query network
        action = LogiChainAction(type="call_tool", tool_name="query_network", arguments={})
        result = await env.step(action)
        print(f"Network:\n{result.observation.tool_result}")
        
        # Assign order
        action = LogiChainAction(
            type="call_tool", 
            tool_name="assign_order",
            arguments={"order_id": "O0", "driver_id": "D0"}
        )
        result = await env.step(action)
        print(f"Result: {result.observation.tool_result}")
        
    finally:
        await env.close()
```

### Context Manager

```python
async with LogisticsEnv.from_docker_image("logistics-env:latest") as env:
    result = await env.reset()
    # ... run episode
```

## Project Structure

```
logistics_env/
├── __init__.py              # Module exports (LogisticsEnv)
├── client.py                # LogisticsEnv WebSocket client
├── models.py                # Action and Observation models
├── inference.py             # LLM inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
├── AGENTS.md                # Agent development guide
└── server/
    ├── __init__.py          # Server module exports
    ├── app.py               # FastAPI application
    ├── logistics_environment.py  # Core environment logic
    ├── tools.py             # Tool implementations
    ├── rewards.py           # Reward computation
    ├── network_graph.py     # NetworkX-based network
    ├── Dockerfile           # Container image definition
    ├── requirements.txt     # Python dependencies
    ├── reward_config.yaml   # Reward configuration
    └── task_config.yaml     # Task definitions
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# Or with options
openenv push --repo-id my-org/logistics-env --public
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI
- **API Documentation** at `/docs` - OpenAPI/Swagger
- **Health Check** at `/health`
- **WebSocket** at `/ws` - Persistent sessions

## License

BSD-style license. See LICENSE file for details.
