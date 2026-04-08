# agents.md

<!--toc:start-->

- [agents.md](#agentsmd)
  - [Purpose](#purpose)
  - [Core Architecture](#core-architecture)
  - [Step 1: Define Models](#step-1-define-models)
    - [Guidelines](#guidelines)
  - [Step 2: Implement Environment](#step-2-implement-environment)
    - [Design Rules](#design-rules)
      - [1. Determinism](#1-determinism)
      - [2. Episode Structure](#2-episode-structure)
      - [3. Action Handling](#3-action-handling)
  - [Step 3: Reward Design](#step-3-reward-design)
  - [Step 4: Tasks & Graders](#step-4-tasks-graders)
    - [Grader Requirements](#grader-requirements)
  - [Step 5: FastAPI Server](#step-5-fastapi-server)
  - [Step 6: Client Implementation](#step-6-client-implementation)
  - [Step 7: Dockerization](#step-7-dockerization)
  - [Step 8: Testing](#step-8-testing)
  - [Step 9: Baseline Agent](#step-9-baseline-agent)
  - [Environment Design Principles](#environment-design-principles)
    - [1. Real-World Task](#1-real-world-task)
    - [2. Partial Observability](#2-partial-observability)
    - [3. Sequential Decisions](#3-sequential-decisions)
    - [4. Clear Success Criteria](#4-clear-success-criteria)
  - [Anti-Patterns (Avoid)](#anti-patterns-avoid)
  - [Checklist](#checklist)
  - [Output Expectation](#output-expectation)
  <!--toc:end-->

## Purpose

You are building an **OpenEnv-compatible environment** using the EnvTorch framework.

Your goal is to implement a **real-world task environment** that:

- follows the OpenEnv interface (`reset`, `step`, `state`)
- uses typed models (Action, Observation, State)
- supports deterministic evaluation with meaningful rewards
- can be deployed via Docker and accessed via an HTTP client

---

## Core Architecture

Every environment MUST follow this structure:

```
envs/my_env/
├── __init__.py
├── models.py
├── client.py
├── README.md
└── server/
    ├── __init__.py
    ├── my_environment.py
    ├── app.py
    └── Dockerfile
```

---

## Step 1: Define Models

Create strongly-typed dataclasses:

```python
from dataclasses import dataclass
from openenv.core.env_server import Action, Observation, State

@dataclass
class MyAction(Action):
    action_type: str
    # additional fields depending on environment

@dataclass
class MyObservation(Observation):
    # what the agent sees
    message: str

@dataclass
class MyState(State):
    # internal environment state
    pass
```

### Guidelines

- Keep actions explicit and structured (avoid free-form strings if possible)
- Observations should contain everything needed for decision-making
- State should track hidden/internal variables

---

## Step 2: Implement Environment

File: `server/my_environment.py`

```python
from openenv.core.env_server import Environment

class MyEnvironment(Environment):

    def __init__(self):
        self._state = MyState()

    def reset(self) -> MyObservation:
        # initialize new episode
        return MyObservation(...)

    def step(self, action: MyAction) -> MyObservation:
        # apply action
        # update state
        # compute reward and done
        return MyObservation(...)

    @property
    def state(self) -> MyState:
        return self._state
```

### Design Rules

#### 1. Determinism

- Same input → same output
- Avoid randomness unless seeded
- Required for reliable grading

#### 2. Episode Structure

- `reset()` must fully initialize state
- `step()` increments `step_count`
- terminate with `done=True`

#### 3. Action Handling

- Validate actions
- Handle invalid actions gracefully (penalty, not crash)

---

## Step 3: Reward Design

Reward must be:

- **Dense** (not just final success)
- **Meaningful** (reflect progress)
- **Bounded** (0.0 to 1.0 recommended)

Example:

```python
reward =
    0.4 * correctness +
    0.3 * efficiency +
    0.3 * safety
```

Penalty examples:

- invalid actions
- unnecessary steps
- destructive operations

---

## Step 4: Tasks & Graders

Each environment MUST include **at least 5 tasks** with difficulty range:

| Task | Difficulty | Done Condition | Grader Type |
|------|------------|----------------|-------------|
| Task 1 | Very Easy | time_limit (20) | volume |
| Task 2 | Easy | time_limit (30) | count_based |
| Task 3 | Medium | all_resolved | efficiency |
| Task 4 | Hard | all_resolved | priority |
| Task 5 | Very Hard | all_resolved | throughput |

### Grader Requirements

- **Deterministic**: Same input → same score every time
- **Reproducible**: Uses only episode state (no random seeds)
- **Normalized**: Returns score between **0.0 and 1.0**
- **Floor before scale**: Apply `max(0.0, score)` then `min(1.0, score / denominator)`

```python
# Correct normalization order:
score = weights.get("delivered", 1.0) * delivered + weights.get("failed", -0.5) * failed
score = max(0.0, score)  # Floor first
score = min(1.0, score / denominator)  # Then scale
```

---

## Step 5: FastAPI Server

File: `server/app.py`

```python
from openenv.core.env_server import create_fastapi_app

env = MyEnvironment()
app = create_fastapi_app(env, MyAction, MyObservation)
```

---

## Step 6: Client Implementation

File: `client.py`

```python
from openenv.core import EnvClient, StepResult

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):

    def _step_payload(self, action: MyAction) -> dict:
        return {...}

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(...)

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(...)
```

---

## Step 7: Dockerization

- Use provided base image (`openenv-base`)
- Install dependencies via `requirements.txt`
- Expose port 8000
- Add `/health` endpoint check

---

## Step 8: Testing

Before Docker:

```python
env = MyEnvironment()
obs = env.reset()
obs = env.step(MyAction(...))
```

After Docker:

```python
client = MyEnv.from_docker_image("my-env:latest")
client.reset()
client.step(...)
```

---

## Step 9: Baseline Agent

Provide a simple script that:

- interacts with the environment
- completes all tasks
- outputs reproducible scores

---

## Environment Design Principles

### 1. Real-World Task

Must simulate something humans actually do:

- data processing
- communication
- decision-making

### 2. Partial Observability

Agent should not see full ground truth.

### 3. Sequential Decisions

Multiple steps required to complete task.

### 4. Clear Success Criteria

Well-defined end condition.

---

## Anti-Patterns (Avoid)

- Toy problems (games, trivial tasks)
- Random reward signals
- Non-deterministic grading
- Unstructured actions (e.g., raw text only)
- Single-step environments

---

## Checklist

Before completion, ensure:

- [ ] Models are typed and clear
- [ ] reset/step/state implemented correctly
- [ ] **5 tasks defined** with difficulty range
- [ ] **Deterministic grader** with 0.0-1.0 output
- [ ] Reward is meaningful and dense
- [ ] Dockerfile builds and runs
- [ ] Client works end-to-end
- [ ] Baseline script produces reproducible scores

---

## Output Expectation

You should produce:

1. Full environment implementation
2. Clear task definitions
3. Deterministic grading logic
4. Working Docker setup
5. Usable client API

Do not skip steps. Do not leave placeholders.
