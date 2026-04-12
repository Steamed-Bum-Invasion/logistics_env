"""
Inference Script for Logistics Env
==================================

This script runs inference on the LogiChain environment using an LLM.

Environment Variables (all optional for local dev, required for submission):
    API_KEY        Your API key (or HF_TOKEN or OPENROUTER_API_KEY for local dev)
    API_BASE_URL   The API endpoint for the LLM (defaults to OpenRouter)
    MODEL_NAME     The model identifier to use (defaults to qwen/qwen-2.5-72b-instruct)

For local development:
    1. Copy .env.example to .env
    2. Add your OPENROUTER_API_KEY
    3. Run: python inference.py

For submission:
    The submission system will inject API_KEY and API_BASE_URL automatically.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from logistics_env import LogisticsEnv
from logistics_env.models import LogiChainAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "logistics_env"
MAX_STEPS = 50
TEMPERATURE = 0.7
MAX_TOKENS = 512

TASKS = [
    "speed_run",
    "quick_delivery",
    "on_time_efficiency",
    "deadline_priority",
    "throughput_master",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are managing a logistics delivery fleet. You have access to tools to:
    - Assign orders to drivers
    - Reroute drivers to dropoff locations
    - Delay or escalate orders to extend deadlines
    - Query driver, order, and network status

    Your goal is to deliver as many orders on time as possible.
    Each order has a deadline. If not delivered before the deadline, it fails.
    You receive rewards for successful deliveries and penalties for failures.

    Available tools:
    - assign_order: Assign a pending order to a driver (needs order_id and driver_id)
    - reroute_driver: Reroute a driver to their order's dropoff (needs driver_id)
    - delay_order: Extend an order's deadline by 3 steps (needs order_id)
    - escalate_order: Extend an order's deadline by 5 steps (needs order_id)
    - query_driver: Get driver status (needs driver_id)
    - query_order: Get order status (needs order_id)
    - query_network: Get network topology and traffic

    ALWAYS respond with a JSON object containing the tool call in this format:
    {"tool_name": "<tool>", "arguments": {"<arg1>": "<value1>", ...}}

    Example: {"tool_name": "query_network", "arguments": {}}
    Example: {"tool_name": "assign_order", "arguments": {"order_id": "O0", "driver_id": "D0"}}
    """).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_escaped = action.replace('"', '\\"')
    print(
        f'[STEP] step={step} action="{action_escaped}" reward={reward:.2f} done={done_val} error={error_val}',
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def parse_tool_call(response_text: str) -> Optional[Dict]:
    """Parse the LLM response to extract tool call."""
    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        data = json.loads(text)
        if "tool_name" in data and "arguments" in data:
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"tool_name"[^{}]*"arguments"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"[DEBUG] Could not parse tool call from: {text[:200]}...", flush=True)
    return None


def get_model_action(
    client: OpenAI,
    model_name: str,
    dashboard: str,
    alerts: List[str],
    available_tools: List[str],
    history: List[str],
) -> str:
    """Get action from the LLM based on current state."""
    history_block = "\n".join(history[-6:]) if history else "No actions taken yet."

    prompt = textwrap.dedent(f"""
        Current State:
        {dashboard}

        Alerts: {alerts if alerts else "None"}

        Available tools: {available_tools}

        History:
        {history_block}

        Choose your next action. Respond with a JSON object:
        {{"tool_name": "<tool_name>", "arguments": {{"arg1": "value1", ...}}}}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        if hasattr(completion, "choices") and completion.choices:
            return (completion.choices[0].message.content or "").strip()
        elif isinstance(completion, str):
            return completion.strip()
        else:
            print(f"[DEBUG] Unexpected response type: {type(completion)}", flush=True)
            return '{"tool_name": "query_network", "arguments": {}}'
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return '{"tool_name": "query_network", "arguments": {}}'


async def run_task(client: OpenAI, task_name: str) -> Dict:
    """Run a single task and return results."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        env = await LogisticsEnv.from_docker_image(IMAGE_NAME)
        result = await env.reset(task_name=task_name)
        obs = result.observation

        last_dashboard = getattr(obs, "dashboard_text", "")
        last_alerts = getattr(obs, "alerts", [])
        last_available_tools = getattr(obs, "available_tools", [])

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            response_text = get_model_action(
                client,
                MODEL_NAME,
                last_dashboard,
                last_alerts,
                last_available_tools,
                history,
            )

            tool_call = parse_tool_call(response_text)

            if tool_call:
                action = LogiChainAction(
                    type="call_tool",
                    tool_name=tool_call.get("tool_name", ""),
                    arguments=tool_call.get("arguments", {}),
                )
                action_str = (
                    f'{tool_call.get("tool_name")}({tool_call.get("arguments", {})})'
                )
            else:
                action = LogiChainAction(
                    type="call_tool", tool_name="query_network", arguments={}
                )
                action_str = "query_network({})"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = getattr(obs, "error_msg", None)

            rewards.append(reward)
            steps_taken = step
            last_dashboard = (
                getattr(obs, "tool_result", "") if hasattr(obs, "tool_result") else ""
            )
            last_alerts = getattr(obs, "alerts", [])
            last_available_tools = getattr(obs, "available_tools", [])

            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error
            )

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        max_possible_reward = MAX_STEPS * 1.5
        score = sum(rewards) / max_possible_reward if max_possible_reward > 0 else 0.0
        score = min(max(score, 0.01), 0.99)
        success = score >= 0.1

    except Exception as e:
        print(f"[ERROR] Task {task_name} failed with error: {e}", flush=True)
        import traceback

        traceback.print_exc()
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task": task_name,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


async def main() -> None:
    """Main entry point."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task in TASKS:
        try:
            result = await run_task(client, task)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to run task {task}: {e}", flush=True)
            results.append(
                {
                    "task": task,
                    "success": False,
                    "steps": 0,
                    "score": 0.0,
                    "rewards": [],
                }
            )

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"Task: {r['task']}, Score: {r['score']:.3f}, Success: {r['success']}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
