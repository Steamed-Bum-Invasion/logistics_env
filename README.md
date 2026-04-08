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

- **Drivers** start at hub nodes and can be assigned to pending orders
- **Orders** spawn dynamically with pickup/dropoff locations and deadlines
- **Network** is a hierarchical graph with hubs, warehouses, zones, and customer areas
- **Traffic** conditions change randomly, affecting travel times
- **Goal**: Deliver as many orders as possible before their deadlines expire

## Network Graph

The network is a hierarchical logistics graph with four node types:

```
Level 0: HUB (central distribution)
    │
Level 1: Warehouses (W1, W2, ...)
    │
Level 2: Zones (Z1, Z2, ...)
    │
Level 3: Customers (C1, C2, ...)
```

### Node Types

| Type | ID Pattern | Description |
|------|------------|-------------|
| `hub` | `HUB`, `H2`, `H3`, ... | Central distribution centers |
| `warehouse` | `W1`, `W2`, ... | Regional warehouses |
| `zone` | `Z1`, `Z2`, ... | Delivery zones/districts |
| `customer` | `C1`, `C2`, ... | Customer delivery areas |

### Road Types

| Type | Capacity | Traffic Range | Description |
|------|----------|---------------|-------------|
| `highway` | 40 | 0.8 - 1.1 | High-speed connections between hubs |
| `arterial` | 25 | 0.9 - 1.5 | Main roads connecting major nodes |
| `standard` | 15 | 1.0 - 2.0 | Local roads to customers |

### Pathfinding

The network uses **Dijkstra's algorithm** with traffic-weighted edges:

```
edge_weight = base_distance × traffic_multiplier
```

All shortest paths are pre-computed and cached for performance. When traffic updates occur, the cache is invalidated and recomputed.

## Environment Description

### Simulation Mechanics

| Mechanic | Description | Default |
|----------|-------------|---------|
| **Time advance** | Every `step()` call advances time by 1 tick | - |
| **Driver movement** | Drivers move 1 unit toward destination per step | - |
| **Order deadlines** | Orders fail immediately when `time_step > deadline` | - |
| **New orders** | Probability per step to spawn 1-2 new orders | 20% |
| **Traffic spikes** | Probability per step for traffic multiplier | 15% |
| **Episode end** | When all orders resolved or max steps reached | - |

### Default Network (Backward Compatible)

The default network has 12 nodes and 18 bidirectional roads:

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

### Delivery Workflow

The agent must manage the full delivery workflow from pickup to dropoff:

```
Step 1: assign_order(order_id, driver_id)
        → Driver routes to PICKUP location
        → Order status: "assigned"

Step 2: Wait for driver to arrive at pickup
        → Driver becomes idle at pickup
        → Order status: "in_transit"

Step 3: reroute_driver(driver_id)
        → Driver routes to DROPOFF location
        → Order status: "in_transit"

Step 4: Wait for driver to arrive at dropoff
        → Driver becomes idle at dropoff
        → Order status: "delivered"
```

**Important Notes:**

- **Rerouting is required**: Without calling `reroute_driver`, the driver will remain idle at the pickup and the order will never be delivered.
- **Driver reassignment**: If you assign a new order to a driver who is already on a delivery, they will abandon their current order(s). Abandoned orders revert to "pending" status and must be reassigned.
- **Reassignment penalty**: Reassigning a busy driver incurs a negative shaping reward (-0.1) to discourage abandoning orders.

### Order Status Transitions

| Status | Description | Transitions To |
|--------|-------------|----------------|
| `pending` | Order waiting to be assigned | `assigned` (via assign_order) |
| `assigned` | Driver routing to pickup | `in_transit` (when driver starts moving) |
| `in_transit` | Driver en route | `delivered` (at dropoff) or `failed` (deadline) |
| `delivered` | Successfully delivered | (terminal) |
| `failed` | Deadline exceeded | (terminal) |

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
| Reassignment (abandoning orders) | -0.08 |
| Successful reroute | +0.02 |
| Delay order | -0.02 |
| Escalate order | -0.05 |
| Invalid action | -0.05 |

### No-Progress Penalty

- After 10 steps without progress (delivery or active orders), a penalty of -0.05 is applied per step
- Active orders include both "assigned" and "in_transit" statuses
- Episode ends after 50 consecutive steps without progress

---

## Experiment Configuration

This section describes how to configure experiments with different network topologies, variation modes, and simulation parameters.

### Network Configuration Methods

The environment supports three ways to define the network:

#### 1. Default Network (No Configuration)

```yaml
# task_config.yaml
tasks:
  my_task:
    # No network section = default 12-node network
    simulation:
      num_drivers: 5
      initial_orders: 10
```

#### 2. Procedural Generation

```yaml
# task_config.yaml
tasks:
  my_task:
    network:
      num_hubs: 2
      num_warehouses: 3
      num_zones: 8
      num_customers: 10
      connectivity: 0.4
      max_edge_distance: 15
      variation_mode: fixed
    simulation:
      num_drivers: 8
      initial_orders: 15
```

#### 3. Explicit Network from File

```yaml
# task_config.yaml
tasks:
  my_task:
    network:
      config_file: networks/custom.yaml
      variation_mode: fixed
    simulation:
      num_drivers: 5
      initial_orders: 10
```

### Network Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_hubs` | int | 1 | Number of hub nodes |
| `num_warehouses` | int | 2 | Number of warehouses |
| `num_zones` | int | 5 | Number of delivery zones |
| `num_customers` | int | 4 | Number of customer areas |
| `connectivity` | float | 0.3 | Edge density (0.0-1.0) |
| `max_edge_distance` | int | 12 | Maximum road distance |
| `variation_mode` | str | `fixed` | Network variation mode |
| `config_file` | str | - | Path to explicit network config |

### Variation Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `fixed` | Same network every reset (seed only) | Reproducible experiments, debugging |
| `per_reset` | Different network each reset | Generalization testing, robustness |

**How it works:**

- `fixed`: `network_seed = seed` (same network every episode)
- `per_reset`: `network_seed = seed + reset_count * 1000` (different network each reset)

**Example:**

```yaml
# Fixed network - agent learns specific topology
tasks:
  fixed_experiment:
    network:
      num_hubs: 1
      num_warehouses: 2
      variation_mode: fixed

# Variable network - agent learns general logistics patterns
tasks:
  variable_experiment:
    network:
      num_hubs: 1
      num_warehouses: 2
      variation_mode: per_reset
```

### Simulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_drivers` | int | 5 | Number of drivers |
| `initial_orders` | int | 10 | Orders at episode start |
| `order_spawn_rate` | float | 0.2 | Probability per step to spawn orders |
| `order_deadline_min` | int | 10 | Minimum deadline (steps from creation) |
| `order_deadline_max` | int | 30 | Maximum deadline (steps from creation) |
| `traffic_spike_rate` | float | 0.15 | Probability per step for traffic spike |
| `traffic_spike_multiplier` | float | 1.3 | Traffic multiplier during spike |

### Creating Custom Network Files

Create a YAML file in `server/networks/`:

```yaml
# server/networks/custom.yaml
nodes:
  - id: HUB
    type: hub
    name: Central Hub
    x: 0
    y: 0
  - id: W1
    type: warehouse
    name: Warehouse Alpha
    x: -3
    y: 2
  - id: Z1
    type: zone
    name: North District
    x: -2
    y: 4
  - id: C1
    type: customer
    name: Customer Zone A
    x: -1
    y: 5

edges:
  - from: HUB
    to: W1
    distance: 8
    capacity: 20
    road_type: arterial
  - from: W1
    to: Z1
    distance: 6
    capacity: 15
    road_type: standard
  - from: Z1
    to: C1
    distance: 5
    capacity: 10
    road_type: standard
```

### Example Experiment Configurations

#### Small Network, Easy Task

```yaml
tasks:
  easy_small:
    done_condition:
      type: time_limit
      max_time_steps: 15
    network:
      num_hubs: 1
      num_warehouses: 1
      num_zones: 3
      num_customers: 2
      connectivity: 0.5
      variation_mode: fixed
    simulation:
      num_drivers: 3
      initial_orders: 5
      order_spawn_rate: 0.1
      order_deadline_min: 15
      order_deadline_max: 40
```

#### Large Network, Hard Task

```yaml
tasks:
  hard_large:
    done_condition:
      type: all_resolved
    network:
      num_hubs: 2
      num_warehouses: 4
      num_zones: 12
      num_customers: 20
      connectivity: 0.4
      max_edge_distance: 15
      variation_mode: per_reset
    simulation:
      num_drivers: 12
      initial_orders: 25
      order_spawn_rate: 0.3
      order_deadline_min: 8
      order_deadline_max: 25
      traffic_spike_rate: 0.25
      traffic_spike_multiplier: 1.5
```

#### Tight Deadlines (Stress Test)

```yaml
tasks:
  stress_test:
    done_condition:
      type: time_limit
      max_time_steps: 50
    network:
      config_file: networks/default.yaml
      variation_mode: fixed
    simulation:
      num_drivers: 5
      initial_orders: 15
      order_spawn_rate: 0.35
      order_deadline_min: 5
      order_deadline_max: 15
      traffic_spike_rate: 0.3
```

### Programmatic Network Generation

You can also generate networks programmatically:

```python
from logistics_env.server.network_graph import NetworkGraph

# Procedural generation
network = NetworkGraph.generate(
    seed=42,
    num_hubs=2,
    num_warehouses=3,
    num_zones=8,
    num_customers=10,
    connectivity=0.4,
    max_edge_distance=15
)

# From config file
network = NetworkGraph.from_config("path/to/network.yaml", seed=42)

# From dictionary
network = NetworkGraph.from_dict({
    "num_hubs": 1,
    "num_warehouses": 2,
    "num_zones": 5,
    "num_customers": 4,
    "connectivity": 0.3
}, seed=42)

# Access network properties
print(f"Nodes: {len(network.nodes)}")
print(f"Hubs: {network.get_hub_nodes()}")
print(f"Zones: {network.get_nodes_by_type('zone')}")
```

---

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
docker build -t logistics-env:latest .
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
├── Dockerfile               # Container image definition
└── server/
    ├── __init__.py          # Server module exports
    ├── app.py               # FastAPI application
    ├── logistics_environment.py  # Core environment logic
    ├── tools.py             # Tool implementations
    ├── rewards.py           # Reward computation
    ├── network_graph.py     # NetworkX-based network
    ├── networks/            # Network configuration files
    │   ├── default.yaml     # Default 12-node network
    │   ├── small_city.yaml  # Small 6-node network
    │   └── complex.yaml     # Large 18-node network
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
