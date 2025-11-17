from datasets import Dataset
import re
import random
import torch
import os
from PIL import Image
from tqdm import tqdm
import traceback
import json


def create_branch_generated_dataset(branch_root: str) -> Dataset:
    """
    Create a HuggingFace Dataset from branch-generated trajectories.

    Expected directory structure under `branch_root`:
        branch_root/
            <branch_id_1>/
                metadata.json
                trajectory.jsonl
                screenshots/
                    step_1.png
                    step_1_replay.png
                    step_2.png
                    step_2_replay.png
                    ...
            <branch_id_2>/
                ...

    Each dataset example corresponds to a single step in a branch trajectory and contains:
        - task_description: overall natural language description of the task
        - branch_dir: absolute path to the branch directory
        - history: concatenated reasoning strings from all previous steps in this branch
        - step: a dict describing the current step, with:
            - step: original integer step id from the trajectory (1-based)
            - is_replay: True if this step comes from the initial replay segment, False otherwise
            - reasoning: optional natural language reasoning for the step
            - action_dict: the high-level action dictionary used by the agent
            - reward: scalar reward
            - done: bool flag
    """
    examples = []

    if not os.path.isdir(branch_root):
        raise ValueError(f"branch_root does not exist or is not a directory: {branch_root}")

    for dir_name in sorted(os.listdir(branch_root)):
        dir_path = os.path.join(branch_root, dir_name)
        if not os.path.isdir(dir_path):
            continue

        metadata_path = os.path.join(dir_path, "metadata.json")
        traj_path = os.path.join(dir_path, "trajectory.jsonl")
        screenshots_dir = os.path.join(dir_path, "screenshots")

        if not (os.path.isfile(metadata_path) and os.path.isfile(traj_path)):
            # Skip directories that do not contain both metadata and trajectory
            continue

        # Load metadata.json to get task descriptions
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[create_branch_generated_dataset] Failed to read {metadata_path}: {e}")
            continue

        task_description = metadata.get("generated_task_description_from_vllm", "")

        num_replay_steps = int(metadata.get("num_replay_steps", 0))

        # First, parse all steps for this branch.
        steps = []
        line_index = 0
        try:
            with open(traj_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[create_branch_generated_dataset] JSON decode error in {traj_path}: {e}")
                        continue
                    line_index += 1

                    # Determine whether this is part of the initial replay segment.
                    # By construction, all replay steps come first in the trajectory.
                    is_replay = line_index <= num_replay_steps

                    # Normalize action_dict:
                    # - use record["action_dict"] if present
                    # - for terminal DONE steps without action_dict, synthesize one
                    # - otherwise keep an empty dict (you can fill these in later)
                    has_action_dict = "action_dict" in record
                    is_terminal_done = (
                        record.get("terminal") is True
                        and record.get("action") == "DONE"
                    )

                    if has_action_dict:
                        action_dict = record.get("action_dict") or {}
                    elif is_terminal_done:
                        action_dict = {"terminal": True, "status": "DONE"}
                    else:
                        action_dict = {}

                    step_id = record.get("step")
                    if step_id is None:
                        step_id = line_index

                    step_entry = {
                        "step": step_id,
                        "is_replay": is_replay,
                        "reasoning": record.get("reasoning", ""),
                        "action_dict": action_dict,
                        "reward": record.get("reward", 0),
                        "done": record.get("done", False),
                    }
                    steps.append(step_entry)
        except Exception as e:
            print(f"[create_branch_generated_dataset] Failed to read {traj_path}: {e}")
            continue

        # Require at least one usable step and at least one screenshot directory
        if not steps:
            continue
        if not os.path.isdir(screenshots_dir):
            print(
                f"[create_branch_generated_dataset] No screenshots directory found for branch {dir_path}, "
                "skipping this branch."
            )
            continue

        # Flatten into one example per step, and precompute textual history for each.
        history_reasonings: list[str] = []
        abs_branch_dir = os.path.abspath(dir_path)
        for step in steps:
            history_text = "\n".join(r for r in history_reasonings if r)
            examples.append(
                {
                    "task_description": task_description,
                    "branch_dir": abs_branch_dir,
                    "history": history_text,
                    "step": step.copy(),
                }
            )

            r = step.get("reasoning", "")
            if r:
                history_reasonings.append(r)

    if not examples:
        raise ValueError(
            f"No valid branch trajectories were found under {branch_root}. "
            "Please check that the directory contains subfolders with metadata.json, "
            "trajectory.jsonl, and a screenshots/ subdirectory."
        )

    print(f"[create_branch_generated_dataset] Loaded {len(examples)} steps from branches under {branch_root}")
    return Dataset.from_list(examples)
