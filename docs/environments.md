# Hackathon Environment Documentation

This document provides comprehensive documentation for the two OpenEnv environments available in the hackathon folder. Both environments are designed to test AI agents on realistic tasks with deterministic grading.

## Table of Contents

1. [Environment Overview](#environment-overview)
2. [Data Cleaning & Transformation Agent](#environment-1-data-cleaning--transformation-agent)
3. [Customer Support Resolution Agent](#environment-2-customer-support-resolution-agent)
4. [Common Usage Patterns](#common-usage-patterns)
5. [Best Practices](#best-practices)

---

## Environment Overview

Both environments follow the OpenEnv pattern and share these characteristics:

| Feature | Description |
|---------|-------------|
| **Interface** | WebSocket + HTTP for agent interactions |
| **Grading** | Deterministic (same input → same score) |
| **Simulation** | Rule-based (no LLM dependency) |
| **Difficulty Levels** | Easy, Medium, Hard |

---

## Environment 1: Data Cleaning & Transformation Agent

### Overview

The **Data Cleaning & Transformation Agent** is a data preprocessing environment where an AI agent learns to clean dirty datasets. The agent receives a "dirty" DataFrame with various data quality issues and must apply cleaning transformations to match a pre-defined "ground truth" clean dataset.

### Core Concept

The environment presents the agent with datasets containing common data quality problems:

- **Missing values** (null, NaN, empty strings)
- **Wrong data types** (numbers stored as strings, dates as text)
- **Duplicate rows** 
- **Inconsistent formatting** (mixed case, whitespace issues)
- **Invalid values** (outliers, impossible dates)

The agent applies cleaning actions, and the system rewards it based on how closely the resulting dataset matches the ground truth.

### Difficulty Levels

| Level | Issues Present | Complexity |
|-------|----------------|------------|
| **Easy** | Missing values, simple type conversion | Single-issue per dataset |
| **Medium** | Inconsistent formats, mixed issues | 2-3 issues per dataset |
| **Hard** | Multi-issue combinations, edge cases | 4+ issues requiring strategy |

### Available Actions

#### 1. drop_column

Remove a column from the dataset.

```python
from hackathon_env.models import DataCleaningAction

action = DataCleaningAction(
    action_type="drop_column",
    column_name="invalid_column"
)
```

**Parameters:**
- `column_name` (str): Name of the column to drop

#### 2. fill_missing

Fill missing values in a column using a specified method.

```python
action = DataCleaningAction(
    action_type="fill_missing",
    column_name="age",
    fill_method="mean",  # or "median", "mode", "forward_fill", "constant", "drop"
    fill_value=None      # Required if fill_method="constant"
)
```

**Parameters:**
- `column_name` (str): Column to fill
- `fill_method` (str): One of `"mean"`, `"median"`, `"mode"`, `"forward_fill"`, `"backward_fill"`, `"constant"`, `"drop"`
- `fill_value` (Any): Value to use when method is `"constant"`

#### 3. convert_type

Convert a column's data type.

```python
action = DataCleaningAction(
    action_type="convert_type",
    column_name="price",
    target_type="float"  # or "int", "string", "datetime", "boolean"
)
```

**Parameters:**
- `column_name` (str): Column to convert
- `target_type` (str): Target data type

#### 4. deduplicate

Remove duplicate rows from the dataset.

```python
action = DataCleaningAction(
    action_type="deduplicate",
    subset=["email", "name"]  # Optional: columns to consider, None = all columns
)
```

**Parameters:**
- `subset` (List[str], optional): Columns to consider for duplicates. If None, considers all columns.

#### 5. submit

Submit the cleaned dataset for grading.

```python
action = DataCleaningAction(
    action_type="submit"
)
```

This action triggers the final evaluation and returns the reward.

### Reward Structure

The reward is a weighted combination of four metrics (total: 100%):

| Component | Weight | Description |
|-----------|--------|-------------|
| **Data Similarity** | 40% | Row-level and cell-level similarity to ground truth |
| **Schema Match** | 30% | Correct column types and structure |
| **Efficiency** | 20% | Fewer steps = higher reward (penalizes over-processing) |
| **Safety** | 10% | No data loss (columns dropped intentionally, not by error) |

**Reward Calculation:**
```
reward = 0.40 * similarity_score + 0.30 * schema_score + 0.20 * efficiency_score + 0.10 * safety_score
```

### Observation Fields

```python
class DataCleaningObservation(Observation):
    current_data: str          # JSON representation of current DataFrame
    columns: List[str]         # Current column names
    dtypes: Dict[str, str]     # Column data types
    missing_counts: Dict[str, int]    # Missing value counts per column
    duplicate_count: int       # Number of duplicate rows
    ground_truth_hint: str     # Description of target (no spoilers!)
    step_info: Dict[str, Any]  # Information about last action result
    reward: float              # Cumulative reward
    done: bool                 # Episode complete
```

### Example Usage

```python
import asyncio
from hackathon_env import DataCleaningEnv

async def clean_dataset():
    # Connect to environment
    async with DataCleaningEnv(base_url="http://localhost:8000") as client:
        # Start new episode with medium difficulty
        result = await client.reset(difficulty="medium")
        
        print(f"Columns: {result.observation.columns}")
        print(f"Missing values: {result.observation.missing_counts}")
        
        # Fill missing ages with median
        await client.step(DataCleaningAction(
            action_type="fill_missing",
            column_name="age",
            fill_method="median"
        ))
        
        # Convert price to float
        await client.step(DataCleaningAction(
            action_type="convert_type",
            column_name="price",
            target_type="float"
        ))
        
        # Remove duplicates
        await client.step(DataCleaningAction(
            action_type="deduplicate",
            subset=["email"]
        ))
        
        # Submit for grading
        result = await client.step(DataCleaningAction(action_type="submit"))
        
        print(f"Final reward: {result.observation.reward}")

asyncio.run(clean_dataset())
```

### Best Practices for Data Cleaning Agent

1. **Inspect before acting**: Use the observation data to understand the issues first
2. **Prioritize safety**: Drop columns only when they're truly useless
3. **Match types early**: Converting types before filling missing can prevent errors
4. **Deduplicate last**: Remove duplicates after other transformations
5. **Test locally**: Use small datasets to verify your cleaning logic

---

## Environment 2: Customer Support Resolution Agent

### Overview

The **Customer Support Resolution Agent** is a conversational AI environment where an agent handles support tickets through natural language interactions. The agent receives customer issues and responds via structured actions, while the system simulates deterministic user responses using rule-based templates.

### Core Concept

The environment presents the agent with support tickets containing various customer issues. The agent must:

1. Understand the customer's problem
2. Respond appropriately or gather more information
3. Resolve the issue or escalate when necessary

The system uses **rule-based user response templates** (NOT LLMs) to ensure deterministic grading—the same initial state always produces the same outcome.

### Difficulty Levels

| Level | Scenario Type | Complexity |
|-------|---------------|------------|
| **Easy** | FAQ questions, simple requests | Direct resolution possible |
| **Medium** | Multi-turn conversations | Requires clarification |
| **Hard** | Ambiguous issues, escalation required | Complex reasoning needed |

**Scenario Categories:**
- Easy: Password reset, shipping status, return policy
- Medium: Billing disputes, product issues, account problems
- Hard: Multi-issue tickets, angry customers, technical escalations

### Available Actions

#### 1. reply

Send a message to the customer.

```python
from hackathon_env.models import SupportAction

action = SupportAction(
    action_type="reply",
    message="I can help you with that. Could you provide your order number?",
    intent="request_info"  # Optional: classify your response
)
```

**Parameters:**
- `message` (str): The response text to send to the customer
- `intent` (str, optional): Classification of the response (e.g., "answer", "question", "apology", "confirmation")

#### 2. request_info

Ask the customer for more information.

```python
action = SupportAction(
    action_type="request_info",
    required_fields=["order_number", "email"],
    context="Need order number to look up shipping status"
)
```

**Parameters:**
- `required_fields` (List[str]): Information needed from customer
- `context` (str): Why this information is needed

#### 3. escalate

Escalate the ticket to a human agent.

```python
action = SupportAction(
    action_type="escalate",
    reason="Complex technical issue beyond agent capability",
    priority="high"  # or "medium", "low"
)
```

**Parameters:**
- `reason` (str): Justification for escalation
- `priority` (str): Escalation priority level

#### 4. close

Close the ticket as resolved.

```python
action = SupportAction(
    action_type="close",
    resolution="refund_processed",
    summary="Customer refunded for order #12345"
)
```

**Parameters:**
- `resolution` (str): Resolution type (e.g., "resolved", "refund", "replacement", "information_provided")
- `summary` (str): Brief summary of resolution

### Reward Structure

The reward is weighted across four dimensions (total: 100%):

| Component | Weight | Description |
|-----------|--------|-------------|
| **Intent Accuracy** | 30% | Correctly identifying customer intent |
| **Resolution Correctness** | 30% | Choosing the right solution path |
| **Efficiency** | 20% | Minimum necessary steps to resolution |
| **Satisfaction** | 20% | Customer sentiment at resolution |

**Intent Categories:**
- `password_reset`, `billing`, `shipping`, `refund`, `technical`, `account`, `faq`

**Resolution Types:**
- `resolved`, `refund`, `replacement`, `escalated`, `information_provided`

### Observation Fields

```python
class SupportObservation(Observation):
    customer_issue: str           # Original customer message
    conversation_history: List[Dict[str, str]]  # Previous messages
    detected_intent: str          # Inferred customer intent
    customer_sentiment: str       # Current sentiment (positive/neutral/negative)
    required_info: List[str]      # Missing information for resolution
    ticket_status: str            # open, pending_customer, pending_internal, resolved, escalated
    available_actions: List[str]  # Actions currently valid
    response_template: str        # Simulated user response (rule-based)
    reward: float                 # Cumulative reward
    done: bool                    # Episode complete
```

### Example Usage

```python
import asyncio
from hackathon_env import SupportEnv

async def handle_ticket():
    async with SupportEnv(base_url="http://localhost:8000") as client:
        # Start new episode with hard difficulty
        result = await client.reset(difficulty="hard")
        
        print(f"Customer: {result.observation.customer_issue}")
        print(f"Intent: {result.observation.detected_intent}")
        
        # Reply to customer
        result = await client.step(SupportAction(
            action_type="reply",
            message="I'm sorry to hear you're experiencing issues. Let me look into this for you.",
            intent="empathy"
        ))
        
        print(f"Customer responds: {result.observation.response_template}")
        
        # Request more info if needed
        if result.observation.required_info:
            result = await client.step(SupportAction(
                action_type="request_info",
                required_fields=result.observation.required_info
            ))
        
        # Handle response and resolve
        result = await client.step(SupportAction(
            action_type="close",
            resolution="refund",
            summary="Processed refund for damaged item"
        ))
        
        print(f"Final reward: {result.observation.reward}")

asyncio.run(handle_ticket())
```

### Determinism Guarantee

The environment ensures **same input → same score** through:

1. **Rule-based response templates**: Pre-defined response patterns based on customer issue + agent action combinations
2. **Fixed intent classification**: Deterministic intent detection based on keyword matching
3. **Deterministic sentiment**: Sentiment changes follow fixed rules (e.g., "refund approved" → positive)
4. **No randomness in escalation**: Escalation decisions are deterministic based on issue complexity

### Best Practices for Support Agent

1. **Acknowledge first**: Always acknowledge the customer's issue before attempting to solve
2. **Gather complete info**: Request all required information before resolving
3. **Match intent correctly**: Ensure your understanding matches the customer's need
4. **Escalate appropriately**: Don't hesitate to escalate when the issue is beyond capability
5. **Provide closure**: Always close with a clear resolution summary

---

## Common Usage Patterns

### Synchronous Client Usage

```python
from hackathon_env import create_sync_client

# For Data Cleaning
with create_sync_client("data_cleaning") as client:
    result = client.reset(difficulty="easy")
    result = client.step(action)
    print(result.observation.reward)

# For Customer Support  
with create_sync_client("customer_support") as client:
    result = client.reset(difficulty="medium")
    result = client.step(action)
    print(result.observation.done)
```

### Async Client Usage

```python
import asyncio
from hackathon_env import create_client

async def main():
    async with create_client("data_cleaning") as client:
        result = await client.reset(difficulty="hard")
        result = await client.step(action)
        print(result.observation.reward)

asyncio.run(main())
```

### Docker Deployment

```python
from hackathon_env import DataCleaningEnv, SupportEnv

# Automatically start container
client = DataCleaningEnv.from_docker_image("hackathon_env:data_cleaning-latest")
try:
    result = client.reset(difficulty="medium")
finally:
    client.close()
```

---

## Best Practices

### General Guidelines

1. **Always reset before starting**: Initialize the environment with `reset(difficulty=...)`
2. **Check done flag**: Stop when `observation.done=True` to avoid unnecessary steps
3. **Use type hints**: Leverage Pydantic models for validation
4. **Handle errors gracefully**: Check `observation.error` if present

### Data Cleaning Agent

- Inspect `missing_counts` before filling
- Check `dtypes` to understand current schema
- Use `ground_truth_hint` to understand goals without spoilers
- Prefer `forward_fill` over `constant` for time series

### Customer Support Agent

- Review `conversation_history` for context
- Check `detected_intent` matches your understanding
- Use `required_info` to guide information gathering
- Monitor `customer_sentiment` for satisfaction scoring

---

## API Reference

### DataCleaningAction

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | str | Yes | One of: drop_column, fill_missing, convert_type, deduplicate, submit |
| `column_name` | str | For column actions | Target column |
| `fill_method` | str | For fill_missing | fill method |
| `fill_value` | Any | For constant fill | Fill value |
| `target_type` | str | For convert_type | Target data type |
| `subset` | List[str] | For deduplicate | Columns to consider |

### SupportAction

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action_type` | str | Yes | One of: reply, request_info, escalate, close |
| `message` | str | For reply | Response text |
| `intent` | str | Optional | Response classification |
| `required_fields` | List[str] | For request_info | Info needed |
| `reason` | str | For escalate | Escalation justification |
| `priority` | str | For escalation | Priority level |
| `resolution` | str | For close | Resolution type |
| `summary` | str | For close | Resolution summary |

---

## Troubleshooting

### Common Issues

**Connection refused**: Ensure the server is running (`uv run python -m server.app`)

**Invalid action**: Check `observation.available_actions` for valid options

**Reward is None**: Ensure you call `submit` (data cleaning) or `close` (support)

**Determinism test fails**: Verify no randomness in action selection

---

## Additional Resources

- OpenEnv Core Documentation: `/home/dhruv/code/personal/OpenEnv/src/openenv/`
- Example Environments: `/home/dhruv/code/personal/OpenEnv/envs/`
- Agent Guidelines: See `AGENT.md` in this directory
