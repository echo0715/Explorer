import argparse
import json
import os
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    MllamaForConditionalGeneration,
)

from PIL import Image
import re
from train_utils import create_branch_generated_dataset

random.seed(123937)

# suggested deepspeed config
DS_CONFIG_DICT = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "round_robin_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        # Enable ZeRO-3 parameter CPU offloading to reduce GPU memory usage
        # while keeping the optimizer on GPU to avoid compiling DeepSpeed CPUAdam,
        # which requires a newer GCC than is available on this cluster.
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
}

def create_model(model_name_or_path, use_flash_attention=False, cache_dir=None):
    """
    Create a LLaMA Vision model for training.
    
    Args:
        model_name_or_path: Path to the pretrained model or HuggingFace model ID
            (e.g., meta-llama/Llama-3.2-11B-Vision-Instruct)
        use_flash_attention: Whether to use Flash Attention 2
        cache_dir: Directory to cache downloaded models
    """
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )

    return model


def build_system_prompt(coordinate_type="relative", processed_width=1000, processed_height=1000):
    """
    Build the system prompt exactly as qwen3vl_agent.py does, including the tools_def JSON.
    """
    description_prompt_lines = [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        (
            f"* The screen's resolution is {processed_width}x{processed_height}."
            if coordinate_type == "absolute"
            else "* The screen's resolution is 1000x1000."
        ),
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
    description_prompt = "\n".join(description_prompt_lines)

    action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor from a specified start (x, y) coordinate to a target (x, y) coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """

    tools_def = {
        "type": "function", 
        "function": {
            "name_for_human": "computer_use", 
            "name": "computer_use", 
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": action_description_prompt,
                        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", 
                                 "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], 
                        "type": "string"
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"}, 
                    "text": {"description": "Required only by `action=type`.", "type": "string"}, 
                    "coordinate": {"description": "The x,y target coordinates for mouse actions.", "type": "array"}, 
                    "start_coordinate": {"description": "The x,y starting coordinates for drag actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"}, 
                    "time": {"description": "The seconds to wait.", "type": "number"}, 
                    "status": {
                        "description": "The status of the task.", 
                        "type": "string", 
                        "enum": ["success", "failure"]
                    }
                }, 
                "required": ["action"], 
                "type": "object"
            }, 
            "args_format": "Format the arguments as a JSON object."
        }
    }

    system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Reasoning: a short reasoning describe the thinking and the action to take in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Reasoning, <tool_call>.
- Be brief: one sentence for Reasoning.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

    return system_prompt


class BranchGeneratedLlamaCollator:
    """
    Data collator for training LLaMA Vision models on branch-generated 
    trajectories, one target step per example, using multi-step, multi-image chat histories.

    Expects each dataset example to have the following fields (as created by
    `create_branch_generated_dataset`):
        - task_description: overall task description for the branch
        - branch_dir: absolute path to the branch directory
        - history: concatenated reasoning strings from all previous steps in this branch
        - step: a dict describing the current step, with:
            - step: integer step id (1-based)
            - is_replay: whether this is a replay step
            - reasoning: optional natural language reasoning
            - action_proposal: optional short imperative description of the action
            - action_dict: high-level action dictionary
        - all_steps: list of dicts for all steps in this branch (same schema as `step`)
        - current_step_idx: index into `all_steps` for the current step
    """

    def __init__(self, args, processor, max_steps=1, dual_training_types=True):
        self.max_steps = max_steps
        self.processor = processor
        self.args = args
        self.dual_training_types = dual_training_types  # Enable both training types
        # Counter to control how often we print input/output for debugging
        self._call_count = 0

    def _build_tool_call_from_action_dict(self, step):
        """
        Convert a high-level action_dict from the dataset into a tool-call JSON
        compatible with LLaMA agent format, of the form:
            {"name": "computer_use", "arguments": {...}}

        Note: Training data coordinates are in 1280x720 pixel space. At runtime,
        the agent with coordinate_type="relative" expects coordinates on a
        0..999 grid in both x and y, which it then scales to the actual screen
        size. Here we convert 1280x720 pixel coordinates into this 0..999
        relative grid so that training and inference use the same convention.
        """
        action_dict = step.get("action_dict") or {}

        # Terminal / DONE steps
        if action_dict.get("terminal") is True:
            status_raw = str(action_dict.get("status", "")).lower()
            if status_raw in ("done", "success"):
                status = "success"
            elif status_raw in ("fail", "failure"):
                status = "failure"
            else:
                status = "success"
            return {
                "name": "computer_use",
                "arguments": {
                    "action": "terminate",
                    "status": status,
                },
            }

        input_dict = action_dict.get("input") or {}
        raw_action_type = input_dict.get("action")
        if not raw_action_type:
            return None

        # Map triple_click → double_click since agent uses double_click
        if raw_action_type == "triple_click":
            action_type = "double_click"
        else:
            action_type = raw_action_type

        arguments = {"action": action_type}

        # Helper function to scale coordinates from 1280x720 absolute pixels
        # into a 0..999 relative integer grid, matching the agent when
        # coordinate_type == "relative".
        def scale_coordinate_to_relative(coord):
            """
            Scale coordinate from 1280x720 pixel space into 0..999 relative
            integer space.

            The agent's parse_response() assumes that for relative coordinates:
                x_screen = x_rel * (original_width / 999)
                y_screen = y_rel * (original_height / 999)
            So here we normalize the recorded 1280x720 coordinates into that
            0..999 range and then round to integers before returning, so the
            model always sees integer coordinates.
            """
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                base_w, base_h = 1280.0, 720.0
                try:
                    x = float(coord[0])
                    y = float(coord[1])
                except Exception:
                    return coord

                # Clamp to the original screen bounds
                x = max(0.0, min(base_w, x))
                y = max(0.0, min(base_h, y))

                # Scale to 0..999 and round to nearest integer
                x_rel = x / base_w * 999.0
                y_rel = y / base_h * 999.0

                x_int = int(round(x_rel))
                y_int = int(round(y_rel))

                # Ensure final coordinates are in-bounds 0..999
                x_int = max(0, min(999, x_int))
                y_int = max(0, min(999, y_int))

                return [x_int, y_int]

            return coord

        # Mouse-based actions with coordinates
        if action_type in (
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "mouse_move",
            "left_click_drag",
        ):
            # For drag actions, we want to preserve both the start and end
            # coordinates so the model learns to move to the start and then
            # drag to the end, matching the pyautogui sequence in the data.
            if action_type == "left_click_drag":
                start_coord = input_dict.get("start_coordinate")
                end_coord = input_dict.get("coordinate")

                if isinstance(start_coord, (list, tuple)) and len(start_coord) == 2:
                    arguments["start_coordinate"] = scale_coordinate_to_relative(start_coord)

                # If no explicit end_coord is provided, fall back to start_coord
                # so that we still have a valid target.
                if isinstance(end_coord, (list, tuple)) and len(end_coord) == 2:
                    arguments["coordinate"] = scale_coordinate_to_relative(end_coord)
                else:
                    print("No end coordinate provided!!!!!!!!!")
            else:
                coord = input_dict.get("coordinate")
                if coord is None:
                    # Some trajectories may only store start_coordinate for
                    # certain mouse actions; fall back to that if present.
                    coord = input_dict.get("start_coordinate")
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    # Scale from 1280x720 absolute pixels into 0..999 relative
                    # coordinates so that training matches agent with
                    # coordinate_type="relative".
                    arguments["coordinate"] = scale_coordinate_to_relative(coord)

            if action_type == "left_click_drag":
                duration = input_dict.get("duration", 0.5)
                # Some trajectories may explicitly store duration as null/None.
                # In that case, or if casting fails, fall back to a sane default.
                if duration is None:
                    duration = 0.5
                try:
                    arguments["duration"] = float(duration)
                except Exception:
                    arguments["duration"] = 0.5

        elif action_type == "type":
            text = input_dict.get("text", "")
            arguments["text"] = str(text)

        elif action_type == "key":
            keys = input_dict.get("keys")
            if not keys:
                # Many trajectories encode key combos as a single string like "ctrl+c"
                key_text = input_dict.get("text", "")
                if isinstance(key_text, str) and key_text:
                    keys = [k.strip() for k in key_text.split("+") if k.strip()]
            if isinstance(keys, str):
                keys = [keys]
            keys = keys or []
            arguments["keys"] = [str(k) for k in keys]

        elif action_type == "scroll":
            amount = input_dict.get("scroll_amount")
            if amount is None:
                amount = input_dict.get("pixels", 0)
            try:
                amount = int(amount)
            except Exception:
                amount = 0
            direction = str(input_dict.get("scroll_direction", "")).lower()
            if direction == "down":
                amount = -abs(amount)
            elif direction == "up":
                amount = abs(amount)
            arguments["pixels"] = amount

        elif action_type == "wait":
            time_val = input_dict.get("time", None)
            if time_val is not None:
                try:
                    arguments["time"] = float(time_val)
                except Exception:
                    pass

        return {
            "name": "computer_use",
            "arguments": arguments,
        }

    def __call__(self, data):
        # Increment call counter for optional debug printing
        self._call_count += 1

        assert (
            len(data) == 1
        ), f"BranchGeneratedLlamaCollator only supports batch_size == 1, got {len(data)}"
        example = data[0]

        overall_task = example["task_description"]
        # Text-only history from the dataset (kept for backward compatibility).
        text_history = example.get("history", "")

        all_steps = example.get("all_steps", None)
        current_step_idx = example.get("current_step_idx", None)

        # Build system prompt exactly as the agent does
        # Training uses 1920x1080 images, coordinate_type defaults to "relative"
        system_prompt_text = build_system_prompt(
            coordinate_type="relative",
            processed_width=1920,
            processed_height=1080
        )

        system_message = {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt_text},
            ],
        }

        branch_dir = example["branch_dir"]

        # Helper to load and resize a screenshot for a given step entry.
        # Note: To predict a step's action, we need to see the screenshot from the PREVIOUS step,
        # since the screenshot shows the state AFTER the previous action was executed.
        def load_step_image(step_entry):
            step_id_local = step_entry.get("step")
            is_replay_local = step_entry.get("is_replay", False)
            
            # For predicting an action, we need the screenshot from the previous step
            if is_replay_local:
                # For replay step N, we need the screenshot from replay step N-1
                prev_step_id = step_id_local - 1
                img_path = os.path.join(
                    branch_dir, "screenshots", f"step_{prev_step_id}_replay.png"
                )
            else:
                # For non-replay step N, we need the screenshot from step N-1
                # If step N is 1, we need the last replay step (step 0 doesn't exist for non-replay)
                prev_step_id = step_id_local - 1
                if prev_step_id < 1:
                    # For step 1, we need to find the last replay step
                    # We'll look for the highest numbered replay step
                    replay_screenshots = []
                    screenshots_dir = os.path.join(branch_dir, "screenshots")
                    if os.path.exists(screenshots_dir):
                        for filename in os.listdir(screenshots_dir):
                            if filename.endswith("_replay.png"):
                                match = re.match(r"step_(\d+)_replay\.png", filename)
                                if match:
                                    replay_screenshots.append(int(match.group(1)))
                    if replay_screenshots:
                        last_replay_step = max(replay_screenshots)
                        img_path = os.path.join(
                            branch_dir, "screenshots", f"step_{last_replay_step}_replay.png"
                        )
                    else:
                        # Fallback: if no replay steps exist, use step_0_replay.png
                        img_path = os.path.join(
                            branch_dir, "screenshots", f"step_0_replay.png"
                        )
                else:
                    img_path = os.path.join(
                        branch_dir, "screenshots", f"step_{prev_step_id}.png"
                    )

            image_local = Image.open(img_path)
            # Use original 1920x1080 resolution without resizing
            return image_local, img_path

        messages = [system_message]
        images = []

        # If we have full trajectory information, build a multi-turn chat history:
        #   System
        #   User (screenshot + instruction at earliest included step)
        #   Assistant (reasoning+action for that step)
        #   ...
        #   User (screenshot for current step, clearly marked as CURRENT)
        use_full_history = all_steps is not None and current_step_idx is not None

        if use_full_history:
            max_past = getattr(self.args, "max_past_screenshots", None)
            if max_past is None:
                max_past = 0
            max_past = max(0, int(max_past))

            # Select a window of past steps to include, ending right before the current step.
            start_idx = max(0, current_step_idx - max_past)

            for idx in range(start_idx, current_step_idx):
                prev_step = all_steps[idx]
                try:
                    prev_image, _ = load_step_image(prev_step)
                except FileNotFoundError:
                    # Skip steps without screenshots.
                    continue

                images.append(prev_image)

                # Build instruction_prompt for the first history step only
                if idx == start_idx:
                    # Build previous actions string for this step
                    prev_actions_for_this_step = []
                    for i in range(idx):
                        if i < len(all_steps):
                            prev_step_i = all_steps[i]
                            prev_desc_i = (
                                prev_step_i.get("reasoning", "")
                                or prev_step_i.get("action_proposal")
                            )
                            prev_actions_for_this_step.append(
                                f"Step {i+1}: {prev_desc_i}"
                            )
                    previous_actions_str = (
                        "\n".join(prev_actions_for_this_step) if prev_actions_for_this_step else "None"
                    )
                    
                    instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {overall_task}

Previous actions:
{previous_actions_str}"""
                    
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": instruction_prompt},
                            ],
                        }
                    )
                else:
                    # Subsequent history steps: just the image
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"}
                            ],
                        }
                    )

                # Assistant response: Action: + <tool_call> format
                # Prefer action_proposal (short imperative) over full reasoning text
                reasoning = prev_step.get("reasoning", "") or prev_step.get(
                    "action_proposal", ""
                )
                tool_call_json = self._build_tool_call_from_action_dict(prev_step)
                
                if tool_call_json:
                    action_line = f"Reasoning: {reasoning}" if reasoning else "Reasoning: Perform action"
                    tool_call_text = f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>"
                    assistant_text = f"{action_line}\n{tool_call_text}"
                else:
                    assistant_text = f"Reasoning: {reasoning}" if reasoning else "Reasoning: Continue"
                
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_text},
                        ],
                    }
                )

        # Each dataset example still corresponds to exactly one (current) step.
        step = example["step"]

        # Load the screenshot for the current step
        # Note: load_step_image already handles loading the previous step's screenshot,
        # since to predict step N's action, we need to see the state from step N-1
        try:
            current_image, image_path = load_step_image(step)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found for current step: {branch_dir}")

        images.append(current_image)

        # Build previous_actions_str for the current step
        previous_actions = []
        if use_full_history and current_step_idx is not None:
            for i in range(current_step_idx):
                if i < len(all_steps):
                    prev_step_i = all_steps[i]
                    prev_desc_i = (
                        prev_step_i.get("reasoning", "")
                        or prev_step_i.get("action_proposal", "")
                    )
                    previous_actions.append(f"Step {i+1}: {prev_desc_i}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        # Build instruction_prompt
        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {overall_task}

Previous actions:
{previous_actions_str}"""

        # Final user turn
        # If this is the first message (no history), include both image and text
        # If we have history, just add the current image (text was in first history message)
        if use_full_history and current_step_idx > 0:
            # We already added instruction_prompt in the first history message,
            # so just add the current screenshot
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}
                    ],
                }
            )
        else:
            # No history or first step: add both image and instruction
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction_prompt},
                    ],
                }
            )

        # Get training type from the example (set during dataset creation)
        # Type 1: predict action_proposal + action
        # Type 2: given action_proposal, predict action only
        training_type = example.get("training_type", "type1")
        use_type_2 = (training_type == "type2")
        
        # For Type 2, add action_proposal to the input prompt
        if use_type_2:
            action_text = step.get("reasoning", "") or step.get("action_proposal", "")
            if action_text:
                # Add action_proposal as part of the user's query
                action_proposal_prompt = f"\nReasoning: {action_text}\n"
                # Add to the last user message
                messages[-1]["content"].append({"type": "text", "text": action_proposal_prompt})
        
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        batch = self.processor(
            text=[prompt], images=[images], padding=True, return_tensors="pt"
        )

        input_ids = [batch["input_ids"]]
        labels = [torch.tensor([-100] * len(batch["input_ids"][0])).unsqueeze(0)]
        
        # LLaMA uses different keys for image features
        # Check what keys are available
        aspect_ratio_ids = batch.get("aspect_ratio_ids", None)
        aspect_ratio_mask = batch.get("aspect_ratio_mask", None)
        cross_attention_mask = batch.get("cross_attention_mask", None)

        # Build supervised target based on training type
        action_text = step.get("reasoning", "") or step.get("action_proposal", "")
        tool_call = self._build_tool_call_from_action_dict(step)

        # Build the textual target
        lines = []
        
        if use_type_2:
            # Type 2: Only output the tool_call (action_proposal was given in input)
            if tool_call and isinstance(tool_call, dict):
                lines.append("<tool_call>")
                lines.append(json.dumps(tool_call, ensure_ascii=False))
                lines.append("</tool_call>")
        else:
            # Type 1: Output both action_proposal and tool_call
            if action_text:
                lines.append(f"Reasoning: {action_text}")
            else:
                # Fallback description if we have no explicit reasoning.
                if tool_call and isinstance(tool_call, dict):
                    args = tool_call.get("arguments", {}) or {}
                    act = args.get("action", "unknown")
                    lines.append(f"Reasoning: Perform {act} action")
                else:
                    lines.append("Reasoning: Decide the next action based on the screenshot.")

            if tool_call and isinstance(tool_call, dict):
                lines.append("<tool_call>")
                lines.append(json.dumps(tool_call, ensure_ascii=False))
                lines.append("</tool_call>")

        # LLaMA uses different EOS tokens
        answer = "\n".join(lines) + "<|eot_id|>"

        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        input_ids.append(answer_input_ids)
        labels.append(answer_input_ids)

        assert "pixel_values" in batch, f"Image not found: {image_path}!!!\n"

        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        pixel_values = batch["pixel_values"]

        attention_mask = torch.ones_like(input_ids)

        batch_out = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }
        
        # Add LLaMA-specific fields if they exist
        if aspect_ratio_ids is not None:
            batch_out["aspect_ratio_ids"] = aspect_ratio_ids
        if aspect_ratio_mask is not None:
            batch_out["aspect_ratio_mask"] = aspect_ratio_mask
        if cross_attention_mask is not None:
            batch_out["cross_attention_mask"] = cross_attention_mask

        # ------------------------------------------------------------------
        # Debug: print model input (prompt) and target output (answer).
        #
        # We only print for the first few batches and then every 100th batch
        # to avoid flooding logs. Text is truncated for readability.
        # ------------------------------------------------------------------
        if self._call_count <= 5 or self._call_count % 100 == 0:
            try:
                training_type_str = "Type 2 (with action_proposal)" if use_type_2 else "Type 1 (predict action_proposal + action)"
                
                # Extract key information
                step_id = step.get("step", "?")
                is_replay = step.get("is_replay", False)
                step_type = "Replay" if is_replay else "Regular"
                num_images_used = len(images)
                
                print("\n" + "╔" + "=" * 98 + "╗")
                print(f"║  TRAINING EXAMPLE #{self._call_count:04d} - {training_type_str:^60s}  ║")
                print("╠" + "=" * 98 + "╣")
                print(f"║  Step: {step_id} ({step_type}) | Images: {num_images_used} | Task: {overall_task[:45]:45s}  ║")
                print("╠" + "=" * 98 + "╣")
                
                # Show the messages structure more clearly
                print("║  MESSAGE STRUCTURE:")
                for i, msg in enumerate(messages):
                    role = msg["role"]
                    content_items = msg["content"]
                    content_types = [item["type"] for item in content_items]
                    print(f"║    [{i}] {role:10s}: {', '.join(content_types)}")
                print("╠" + "=" * 98 + "╣")
                
                # Show the full prompt (input to model)
                print("║  MODEL INPUT PROMPT:")
                print("╠" + "-" * 98 + "╣")
                prompt_lines = prompt.split('\n')
                for line in prompt_lines[:100]:  # Show first 100 lines
                    # Truncate very long lines
                    if len(line) > 96:
                        print(f"║  {line[:93]}...")
                    else:
                        print(f"║  {line:96s}║")
                if len(prompt_lines) > 100:
                    print(f"║  ... [{len(prompt_lines) - 100} more lines omitted] ...")
                
                print("╠" + "=" * 98 + "╣")
                
                # Show the target answer (what model should output)
                print("║  MODEL TARGET OUTPUT:")
                print("╠" + "-" * 98 + "╣")
                answer_lines = answer.split('\n')
                for line in answer_lines:
                    # Truncate very long lines
                    if len(line) > 96:
                        print(f"║  {line[:93]}...")
                    else:
                        print(f"║  {line:96s}║")
                
                print("╠" + "=" * 98 + "╣")
                
                # Show token statistics
                num_input_tokens = input_ids.shape[1]
                num_label_tokens = (labels[0] != -100).sum().item()
                print(f"║  TOKENS: Input={num_input_tokens:5d} | Labels={num_label_tokens:5d} | Images={num_images_used:2d}" + " " * 38 + "║")
                
                print("╚" + "=" * 98 + "╝\n")
                
            except Exception as e:
                # Never break training because of debug printing
                print(f"[Collator Debug] Failed to print input/output: {e}")
                import traceback
                traceback.print_exc()

        return batch_out


class SafeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_token_counts = []  # Track tokens per GPU for current step
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            # Print token usage for this training step
            if "input_ids" in inputs:
                num_input_tokens = inputs["input_ids"].shape[1]
                num_labels = (inputs["labels"] != -100).sum().item()
                num_images = inputs.get("pixel_values", torch.tensor([])).shape[0] if "pixel_values" in inputs else 0
                
                # Track for aggregation across gradient accumulation steps
                self.step_token_counts.append(num_input_tokens)
            
            # Run the standard training step
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                num_input_tokens = inputs["input_ids"].shape[1]
                num_labels = (inputs["labels"] != -100).sum().item()
                print(f"[OOM ERROR] Failed on {num_input_tokens} input tokens, {num_labels} label tokens. Step token history: {self.step_token_counts}")
                self.step_token_counts = []  # Reset
                # Clear the CUDA cache to recover memory
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device=next(model.parameters()).device)
            else:
                raise e  # Re-raise if it's not an OOM error


def main():
    """
    Train LLaMA Vision models on branch-generated trajectories.
    
    Example usage for LLaMA 3.2 11B Vision:
        torchrun --nproc_per_node=4 train_llama.py \\
            --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \\
            --use_flash_attention \\
            --batch_size 16 \\
            --output_dir /path/to/output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name or path to load from (e.g., meta-llama/Llama-3.2-11B-Vision-Instruct or meta-llama/Llama-3.2-90B-Vision-Instruct)",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="/gpfs/radev/home/jw3278/scratch/hf_cache",
        help="Directory to use for Hugging Face cache (models, tokenizers, etc.)",
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="Use Flash Attention"
    )
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 (default True for LLaMA)")
    parser.add_argument(
        "--output_dir", type=str, default="/gpfs/radev/home/jw3278/scratch/llama-vision-train", help="Output directory"
    )
    parser.add_argument(
        "--save-strategy", type=str, default="steps", help="Save strategy"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm"
    )
    parser.add_argument(
        "--tensorboard-logging", action="store_true", help="log to tensorboard"
    )
    parser.add_argument(
        "--use-google-search",
        action="store_true",
        help="add google search in action space and prompt",
    )
    parser.add_argument(
        "--use-nogoto-gs-format",
        action="store_true",
        help="remove gs and goto from prompt",
    )
    parser.add_argument(
        "--branch_generated_root",
        type=str,
        default="/gpfs/radev/home/jw3278/scratch/branch_generated_verified_done",
        help=(
            "Root directory containing branch-generated trajectories "
            "(each subdirectory should contain metadata.json, trajectory.jsonl, and screenshots/). "
            "If provided, this will be used to build the training dataset instead of --train_dir."
        ),
    )
    parser.add_argument(
        "--max_past_screenshots",
        type=int,
        default=2,
        help=(
            "Maximum number of past screenshots (steps) to include in the prompt, "
            "in addition to the current step. Default is 2. "
            "Set to 0 to only use the current screenshot."
        ),
    )
    parser.add_argument(
        "--dual_training_types",
        action="store_true",
        default=True,
        help=(
            "Enable dual training types: "
            "Type 1: instruction + history → predict (action_proposal + action), "
            "Type 2: instruction + history + action_proposal → predict action only. "
            "When enabled, creates BOTH training examples for each step (doubles the dataset size). "
            "Default is True."
        ),
    )
    parser.add_argument(
        "--no_dual_training_types",
        dest="dual_training_types",
        action="store_false",
        help="Disable dual training types and only use Type 1 (original behavior)."
    )

    args = parser.parse_args()

    accelerator = Accelerator()

    # Ensure Hugging Face uses the requested cache directory.
    if args.hf_cache_dir:
        os.makedirs(args.hf_cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", args.hf_cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", args.hf_cache_dir)

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=args.hf_cache_dir,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            cache_dir=args.hf_cache_dir,
        )

    # Build training dataset from branch-generated trajectories
    if not args.branch_generated_root:
        raise ValueError(
            "You must provide --branch_generated_root pointing to the branch_generated directory."
        )
    train_dataset = create_branch_generated_dataset(
        args.branch_generated_root,
        dual_training_types=args.dual_training_types
    )

    # Shuffle the dataset to ensure Type 1 and Type 2 examples of the same step
    # are not adjacent. This prevents the model from memorizing consecutive patterns.
    train_dataset = train_dataset.shuffle(seed=42)

    print("train_dataset:", train_dataset)
    print("len(train_dataset):", len(train_dataset))
    print("Dataset shuffled to randomize Type 1 and Type 2 examples.")

    import time

    time.sleep(3)

    num_gpus = accelerator.num_processes
    print(f"training on {num_gpus} GPUs")
    assert (
        args.batch_size % num_gpus == 0
    ), "Batch size must be divisible by the number of GPUs"
    gradient_accumulation_steps = args.batch_size // num_gpus
    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,  # Set to False for LLaMA (more efficient)
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # NOTE currently only supports batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=20,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy=args.save_strategy,
        save_steps=20,
        save_total_limit=5 if args.save_strategy == "steps" else None,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to="tensorboard" if args.tensorboard_logging else "none",
        deepspeed=DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
    )

    data_collator = BranchGeneratedLlamaCollator(
        args, processor, dual_training_types=args.dual_training_types
    )

    # Save a "checkpoint-0" copy of the original (pre-finetune) model & processor
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        pre_ft_ckpt_dir = out_path / "checkpoint-0"
        pre_ft_ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save the base model and processor before any training updates
        model.save_pretrained(pre_ft_ckpt_dir)
        processor.save_pretrained(pre_ft_ckpt_dir)
    accelerator.wait_for_everyone()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = model.to(f"cuda:{local_rank}")

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

