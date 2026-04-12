"""
Inference Script for Logistics Env
==================================

This script runs inference on the LogiChain environment using an LLM.

Requirements:
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM (provided by submission system).
    API_KEY        Your API key (provided by submission system).
    MODEL_NAME     The model identifier to use for inference.

The inference script must be named `inference.py` and placed in the root directory.
Participants must use OpenAI Client for all LLM calls.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from logistics_env import LogisticsEnv
from logistics_env.models import LogiChainAction

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


def check_docker_image(image_name: str) -> bool:
    """Check if Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[DEBUG] Failed to check Docker image: {e}", flush=True)
        return False


def build_docker_image(image_name: str) -> bool:
    """Build Docker image if it doesn't exist."""
    try:
        print(f"[DEBUG] Building Docker image: {image_name}", flush=True)
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "."],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[DEBUG] Docker build failed: {result.stderr}", flush=True)
            return False
        print(f"[DEBUG] Docker image built successfully", flush=True)
        return True
    except Exception as e:
        print(f"[DEBUG] Failed to build Docker image: {e}", flush=True)
        return False


async def run_task(client: OpenAI, task_name: str, model_name: str, image_name: str) -> Dict:
    """Run a single task and return results."""
    log_start(task=task_name, env=BENCHMARK, model=model_name)

    env = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Check if Docker image exists, build if needed
        if not check_docker_image(image_name):
            print(f"[DEBUG] Docker image not found, attempting to build...", flush=True)
            if not build_docker_image(image_name):
                print(f"[ERROR] Failed to build Docker image", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return {
                    "task": task_name,
                    "success": False,
                    "steps": 0,
                    "score": 0.0,
                    "rewards": [],
                }

        # Create environment with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                env = await LogisticsEnv.from_docker_image(image_name)
                break
            except Exception as e:
                print(
                    f"[DEBUG] Attempt {attempt + 1} failed to create environment: {e}",
                    flush=True,
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise

        result = await env.reset(task_name=task_name)
        obs = result.observation

        last_dashboard = getattr(obs, "dashboard_text", "")
        last_alerts = getattr(obs, "alerts", [])
        last_available_tools = getattr(obs, "available_tools", [])
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                response_text = get_model_action(
                    client, model_name, last_dashboard, last_alerts, last_available_tools, history
                )

                tool_call = parse_tool_call(response_text)

                if tool_call:
                    action = LogiChainAction(
                        type="call_tool",
                        tool_name=tool_call.get("tool_name", ""),
                        arguments=tool_call.get("arguments", {}),
                    )
                    action_str = f'{tool_call.get("tool_name")}({tool_call.get("arguments", {})})'
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
                last_reward = reward
                last_dashboard = (
                    getattr(obs, "tool_result", "")
                    if hasattr(obs, "tool_result")
                    else ""
                )
                last_alerts = getattr(obs, "alerts", [])
                last_available_tools = getattr(obs, "available_tools", [])

                log_step(
                    step=step, action=action_str, reward=reward, done=done, error=error
                )

                history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

                if done:
                    break

            except Exception as e:
                print(f"[DEBUG] Step {step} failed: {e}", flush=True)
                rewards.append(-0.05)
                steps_taken = step
                log_step(
                    step=step, action="error", reward=-0.05, done=False, error=str(e)
                )

        max_possible_reward = MAX_STEPS * 1.5
        score = sum(rewards) / max_possible_reward if max_possible_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
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
    # Read environment variables at runtime (not at import time)
    API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
    API_BASE_URL = os.environ.get("API_BASE_URL")
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME", "logistics-env:latest")

    if not API_KEY:
        print("ERROR: API_KEY must be set", flush=True)
        return

    if not API_BASE_URL:
        print("ERROR: API_BASE_URL must be set", flush=True)
        return

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME must be set", flush=True)
        return

    print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME: {MODEL_NAME}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task in TASKS:
        try:
            result = await run_task(client, task, MODEL_NAME, LOCAL_IMAGE_NAME)
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
