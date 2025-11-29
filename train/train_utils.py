from datasets import Dataset
import re
import random
import torch
import os
from PIL import Image
from tqdm import tqdm
import traceback
import json
from typing import Optional


def create_branch_generated_dataset(
    branch_root: str,
    dual_training_types: bool = True,
    half_verified_root: Optional[str] = None,
) -> Dataset:
    """
    Create a HuggingFace Dataset from branch-generated trajectories.

    Args:
        branch_root: Root directory containing branch subdirectories
        dual_training_types: If True, creates both Type 1 and Type 2 examples for each step,
                           doubling the dataset size. If False, only creates Type 1 examples.

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
            - action_proposal: optional short imperative description of the action
            - action_dict: the high-level action dictionary used by the agent
            - reward: scalar reward
            - done: bool flag
        - all_steps: list of dicts for all steps in this branch (same schema as `step`)
        - current_step_idx: integer index into `all_steps` for the current step
        - training_type: "type1" or "type2" indicating which training format to use
    """
    examples = []

    if not os.path.isdir(branch_root):
        raise ValueError(f"branch_root does not exist or is not a directory: {branch_root}")

    if half_verified_root is not None and not os.path.isdir(half_verified_root):
        raise ValueError(
            f"half_verified_root was provided but does not exist or is not a directory: {half_verified_root}"
        )

    # ------------------------------------------------------------------
    # First pass: collect all branches and their metadata so we can
    # determine, for LibreOffice domains, which branches should KEEP
    # replay steps and which should DROP them.
    #
    # We treat:
    #   - entries from `branch_root` as source == "main"
    #   - entries from `half_verified_root` (if provided) as source == "half_verified"
    #
    # The LibreOffice "max branch keeps replay" logic only looks at
    # branches from the main root so that adding half-verified data
    # does not change the behavior of existing training branches.
    # ------------------------------------------------------------------
    branch_infos = []

    def _collect_branch_infos(root: str, source: str) -> None:
        for dir_name in sorted(os.listdir(root)):
            dir_path = os.path.join(root, dir_name)
            if not os.path.isdir(dir_path):
                continue

            metadata_path = os.path.join(dir_path, "metadata.json")
            traj_path = os.path.join(dir_path, "trajectory.jsonl")
            screenshots_dir = os.path.join(dir_path, "screenshots")

            if not (os.path.isfile(metadata_path) and os.path.isfile(traj_path)):
                # Skip directories that do not contain both metadata and trajectory
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[create_branch_generated_dataset] Failed to read {metadata_path}: {e}")
                continue

            branch_infos.append(
                {
                    "dir_path": dir_path,
                    "metadata_path": metadata_path,
                    "traj_path": traj_path,
                    "screenshots_dir": screenshots_dir,
                    "metadata": metadata,
                    "source": source,
                }
            )

    # Always collect from the main root
    _collect_branch_infos(branch_root, source="main")
    # Optionally, also collect half-verified branches
    if half_verified_root is not None:
        _collect_branch_infos(half_verified_root, source="half_verified")

    # Map from branch dir to whether we should keep replay steps when
    # building the training data. Default is True (keep replays).
    include_replay_for_branch: dict[str, bool] = {}

    # Only apply the "drop replay steps for non-max branches" logic to
    # LibreOffice domains, grouped by original_task_id, and *only* for
    # branches coming from the main root. Half-verified branches are
    # always treated as "drop replay, keep only post-branch steps".
    target_domains = {"libreoffice_calc", "libreoffice_impress", "libreoffice_writer"}
    branches_by_original_id: dict[str, list[dict]] = {}

    for info in branch_infos:
        metadata = info["metadata"]
        if info.get("source") != "main":
            # Only main-root branches participate in the max-branch selection.
            continue
        domain = metadata.get("domain")
        if domain not in target_domains:
            continue

        original_task_id = metadata.get("original_task_id")
        if not original_task_id:
            continue

        branch_after_step_raw = metadata.get("branch_after_step")
        branch_after_step_int = int(branch_after_step_raw)

        info["branch_after_step_int"] = branch_after_step_int
        # Group branches that share the same original_task_id
        branches_by_original_id.setdefault(str(original_task_id), []).append(info)

    for _task_id, infos in branches_by_original_id.items():
        max_branch_step = max(i["branch_after_step_int"] for i in infos)
        for info in infos:
            dir_path = info["dir_path"]
            include_replay_for_branch[dir_path] = info["branch_after_step_int"] == max_branch_step

    # ------------------------------------------------------------------
    # Second pass: actually build per-step training examples, using the
    # include_replay_for_branch map to optionally drop replay steps for
    # non-max LibreOffice branches.
    # ------------------------------------------------------------------
    for info in branch_infos:
        dir_path = info["dir_path"]
        metadata = info["metadata"]
        traj_path = info["traj_path"]
        screenshots_dir = info["screenshots_dir"]

        # For LibreOffice domains that are not the max-branch for a given
        # original_task_id, we drop replay steps from training and history.
        # Look up this flag before deciding which task description to use.
        include_replay = include_replay_for_branch.get(dir_path, True)

        source = info.get("source", "main")
        domain = metadata.get("domain")

        # Half-verified branches:
        #   - always drop replay steps (use only post-branch steps)
        #   - always prefer the human-edited `new_task_description`
        if source == "half_verified":
            include_replay = False
            task_description = metadata.get(
                "new_task_description",
                metadata.get("generated_task_description_from_vllm", ""),
            )
        else:
            # For non-max LibreOffice branches (where we drop replay steps),
            # use the human-edited `new_task_description` instead of the
            # VLLM-generated one. For all other branches, keep the original
            # behavior.
            if not include_replay and domain in target_domains:
                task_description = metadata.get(
                    "new_task_description",
                    metadata.get("generated_task_description_from_vllm", ""),
                )
            else:
                task_description = metadata.get("generated_task_description_from_vllm", "")
        num_replay_steps = int(metadata.get("num_replay_steps", 0))

        # For replay steps, we may have additional metadata describing how the
        # reasoning was backfilled (e.g., "candidate_match" vs "fallback_direct").
        # We only want to create supervised training targets from replay steps
        # whose backfill method is "candidate_match", but we still want ALL
        # replay steps to appear in the trajectory/history so that later steps
        # can see their reasoning as context.
        replay_reasoning_backfill = metadata.get("replay_reasoning_backfill", {}) or {}
        replay_updated_steps = replay_reasoning_backfill.get("updated_steps", {}) or {}
        # Map from 1-based replay step index -> backfill method string
        replay_method_by_index: dict[int, str] = {}
        for k, v in replay_updated_steps.items():
            try:
                idx_int = int(k)
            except (TypeError, ValueError):
                continue
            method = v.get("method")
            if isinstance(method, str):
                replay_method_by_index[idx_int] = method

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

                    # For non-max LibreOffice branches we skip replay steps entirely:
                    #   - they will not get supervised targets
                    #   - they will not appear in `all_steps` or textual history
                    # The collator will still use the last replay screenshot as
                    # the initial observation via its screenshot-loading logic.
                    if is_replay and not include_replay:
                        continue

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

                    # Decide whether to skip creating training examples for this step.
                    # - We always keep the step in `steps` (so it shows up in
                    #   `all_steps` and its reasoning can contribute to history).
                    # - If skip=True, we will not create supervised training
                    #   examples for this step later on.
                    skip_flag = record.get("skip", False)
                    if is_replay:
                        # For replay steps, if the backfill method is present and
                        # not "candidate_match" (e.g., "fallback_direct"), then we
                        # only want this step for context, not as a supervised
                        # training target.
                        replay_method = replay_method_by_index.get(line_index)
                        if replay_method is not None and replay_method != "candidate_match":
                            skip_flag = True

                    step_entry = {
                        "step": step_id,
                        "is_replay": is_replay,
                        "reasoning": record.get("reasoning", ""),
                        # Short imperative description of the action, if present.
                        # This is preferred over `reasoning` when building the
                        # supervised "Action: ..." target during training.
                        "action_proposal": record.get("action_proposal", ""),
                        "action_dict": action_dict,
                        "reward": record.get("reward", 0),
                        "done": record.get("done", False),
                        # If skip is True, we still include this step in the trajectory
                        # (for context/history), but don't create training examples from it
                        "skip": skip_flag,
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
        num_steps = len(steps)
        for idx, step in enumerate(steps):
            history_text = "\n".join(r for r in history_reasonings if r)
            is_last_step = idx == num_steps - 1

            # For the final step in the trajectory, force a standardized
            # action_proposal so the model always sees a clear terminal message.
            if is_last_step:
                step = step.copy()
                step["action_proposal"] = "The task is completed successfully."
                steps[idx] = step

            # Skip creating training examples for steps marked with skip=True.
            # These steps remain in the trajectory for context (history and screenshots),
            # but we don't generate supervised targets from them.
            should_skip = step.get("skip", False)
            
            if not should_skip:
                # Create training examples for all non-skipped steps
                base_example = {
                    "task_description": task_description,
                    "branch_dir": abs_branch_dir,
                    "history": history_text,
                    "step": step.copy(),
                    # Provide full trajectory and index so the collator can
                    # reconstruct multi-step, multi-image chat histories.
                    "all_steps": steps,
                    "current_step_idx": idx,
                }
                
                # Create Type 1 example: predict action_proposal + action
                example_type1 = base_example.copy()
                example_type1["training_type"] = "type1"
                examples.append(example_type1)
                
                # If dual_training_types is enabled, also create Type 2 example
                if dual_training_types:
                    # Create Type 2 example: given action_proposal, predict action only
                    example_type2 = base_example.copy()
                    example_type2["training_type"] = "type2"
                    examples.append(example_type2)

            # Always add reasoning to history (even for skipped steps)
            # so that later steps have the full context
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
