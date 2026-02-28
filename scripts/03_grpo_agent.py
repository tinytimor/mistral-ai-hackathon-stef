#!/usr/bin/env python3
"""
03_grpo_agent.py — GRPO (Group Relative Policy Optimization) agent training
to teach the model to think → plan → act → reflect with tool use.

This uses TRL's GRPOTrainer with the built-in `tools` parameter for
reinforcement learning with tool-calling feedback.

Runs on RTX 5090 (32GB VRAM).

Prerequisites:
    pip install torch transformers trl peft bitsandbytes datasets accelerate vllm

Usage:
    python scripts/03_grpo_agent.py --model models/ministral-3b-sft --output models/ministral-3b-grpo
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# ─── Tool definitions (must match 01_generate_training_data.py) ──────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_look_at",
            "description": "Make the robot look at a point in 3D space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
                    "duration": {"type": "number", "default": 1.0},
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_express",
            "description": "Make the robot express an emotion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string", "enum": ["happy", "sad", "curious", "surprised", "thinking", "nodding", "shaking_no"]},
                    "intensity": {"type": "number", "default": 0.7},
                },
                "required": ["emotion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_speak",
            "description": "Make the robot speak text aloud.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "language": {"type": "string", "default": "en"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_summary",
            "description": "Retrieve a patient's health summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "include_vitals": {"type": "boolean", "default": True},
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a timed reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "minutes": {"type": "integer"},
                },
                "required": ["message", "minutes"],
            },
        },
    },
]

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

# ─── Training prompts for GRPO ───────────────────────────────────────────────
TRAINING_PROMPTS = [
    "What's the weather like in NYC today?",
    "Look at me and tell me a fun fact.",
    "Search for the latest medical news and share it excitedly.",
    "I feel dizzy, can you help?",
    "Remind me in 15 minutes to take my pills.",
    "What are the side effects of ibuprofen?",
    "Look to your left and describe what you see.",
    "Tell me a joke and laugh about it.",
    "Pull up patient PT-12345's vitals.",
    "What are the symptoms of COVID-19?",
    "Help me plan my exercise routine.",
    "Search for healthy dinner recipes.",
    "Nod if you understand what I'm saying.",
    "What time is it in London?",
    "Express curiosity about my outfit.",
    "What medications interact with aspirin?",
    "Search for mental health resources and share them empathetically.",
    "Look down at the table and read what's there.",
    "Set a reminder for my 3pm meeting.",
    "What are the warning signs of a heart attack?",
    "Compare the benefits of yoga vs pilates.",
    "Search for the nearest pharmacy and tell me the address.",
    "I'm feeling anxious. Can you help me with a breathing exercise?",
    "Look up and tell me about the ceiling.",
    "What's the latest research on intermittent fasting?",
    "Express surprise and say 'wow, that's amazing!'",
    "Summarize the news about healthcare policy changes.",
    "Help me understand my blood test results.",
    "Search for first aid procedures for a minor burn.",
    "Look at me with concern and ask how I'm feeling.",
]


# ─── Reward Functions ────────────────────────────────────────────────────────

def reward_format_correctness(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward for producing valid tool call format."""
    rewards = []
    for completion in completions:
        text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        score = 0.0

        # Check for <tool_call> tags
        if "<tool_call>" in text and "</tool_call>" in text:
            score += 0.3

            # Check if the JSON inside is valid
            tool_calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
            for tc in tool_calls:
                try:
                    parsed = json.loads(tc)
                    if "name" in parsed and "arguments" in parsed:
                        score += 0.3
                        # Check if tool name is valid
                        if parsed["name"] in VALID_TOOL_NAMES:
                            score += 0.4
                except json.JSONDecodeError:
                    pass

        # Also reward think tags (think-plan-act-reflect)
        if "<think>" in text and "</think>" in text:
            score += 0.2

        rewards.append(min(score, 1.0))
    return rewards


def reward_tool_relevance(completions: list[list[dict]], prompts: list[str] = None, **kwargs) -> list[float]:
    """Reward for choosing relevant tools for the prompt."""
    if prompts is None:
        return [0.0] * len(completions)

    rewards = []
    for i, completion in enumerate(completions):
        text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        prompt = prompts[i] if i < len(prompts) else ""
        prompt_lower = prompt.lower()
        score = 0.0

        # Extract tool names used
        tool_calls = re.findall(r'"name":\s*"(\w+)"', text)

        for tool in tool_calls:
            # Check relevance heuristics
            if tool == "search_web" and any(w in prompt_lower for w in ["search", "news", "latest", "what is", "find", "compare", "look up", "weather", "time"]):
                score += 0.5
            elif tool == "robot_look_at" and any(w in prompt_lower for w in ["look", "see", "watch", "down", "up", "left", "right", "at me"]):
                score += 0.5
            elif tool == "robot_express" and any(w in prompt_lower for w in ["feel", "emotion", "happy", "sad", "curious", "nod", "surprise", "concern", "excit"]):
                score += 0.5
            elif tool == "robot_speak" and any(w in prompt_lower for w in ["tell", "say", "share", "describe", "explain"]):
                score += 0.3
            elif tool == "get_patient_summary" and any(w in prompt_lower for w in ["patient", "vitals", "record", "PT-"]):
                score += 0.5
            elif tool == "set_reminder" and any(w in prompt_lower for w in ["remind", "reminder", "minutes", "meeting"]):
                score += 0.5

        rewards.append(min(score, 1.0))
    return rewards


def reward_response_quality(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward for overall response quality — empathy, completeness, professionalism."""
    rewards = []
    for completion in completions:
        text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        score = 0.0

        # Penalize empty or too-short responses
        if len(text) < 20:
            rewards.append(-0.5)
            continue

        # Reward appropriate length (not too short, not too long)
        if 50 < len(text) < 1500:
            score += 0.2

        # Reward empathetic language
        empathy_words = ["understand", "sorry", "help", "care", "feel", "concern", "hope", "important", "safe"]
        if any(w in text.lower() for w in empathy_words):
            score += 0.2

        # Reward professional medical language when appropriate
        medical_words = ["symptoms", "medication", "health", "doctor", "consult", "professional"]
        if any(w in text.lower() for w in medical_words):
            score += 0.1

        # Reward multi-tool usage (shows planning)
        tool_calls = re.findall(r'<tool_call>', text)
        if len(tool_calls) >= 2:
            score += 0.3
        elif len(tool_calls) == 1:
            score += 0.1

        # Penalize repetition
        sentences = text.split(". ")
        if len(sentences) > 1 and len(set(sentences)) < len(sentences) * 0.7:
            score -= 0.3

        rewards.append(min(max(score, -1.0), 1.0))
    return rewards


def reward_thinking_quality(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward for showing think-plan-act-reflect reasoning."""
    rewards = []
    for completion in completions:
        text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        score = 0.0

        # Extract thinking content
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).lower()
            score += 0.2  # Base reward for thinking

            # Reward planning language
            if any(w in thinking for w in ["plan", "first", "then", "need to", "should", "step"]):
                score += 0.2

            # Reward reflection language
            if any(w in thinking for w in ["consider", "might", "could", "best", "appropriate", "because"]):
                score += 0.2

            # Reward tool selection reasoning
            if any(w in thinking for w in ["search", "look", "express", "speak", "remind", "patient"]):
                score += 0.2

            # Penalize too-short thinking
            if len(thinking) < 15:
                score -= 0.1

        rewards.append(min(score, 1.0))
    return rewards


def create_grpo_dataset() -> Dataset:
    """Create a dataset of prompts for GRPO training."""
    # Duplicate prompts for more training data
    expanded = []
    for prompt in TRAINING_PROMPTS:
        expanded.append({"prompt": prompt})
        # Add variations
        expanded.append({"prompt": f"Hey Reachy, {prompt.lower()}"})
        expanded.append({"prompt": f"Please {prompt.lower()}"})

    return Dataset.from_list(expanded)


def main():
    parser = argparse.ArgumentParser(description="GRPO agent training for tool-calling")
    parser.add_argument("--model", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--output", type=str, default="models/ministral-3b-grpo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of GRPO epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (lower for RL)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎯 GRPO Agent Training — Tool-Calling Reinforcement Learning")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ─── Load tokenizer ──────────────────────────────────────────────────
    print(f"\n📦 Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Load model with QLoRA ───────────────────────────────────────────
    print(f"\n🧠 Loading model from {args.model} with QLoRA...")
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=qlora_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ─── LoRA for GRPO (separate adapter from SFT) ──────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,           # Smaller rank for RL fine-tuning
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # ─── Create training dataset ─────────────────────────────────────────
    print("\n📊 Creating GRPO training dataset...")
    dataset = create_grpo_dataset()
    print(f"   Total prompts: {len(dataset)}")

    # ─── GRPO configuration ─────────────────────────────────────────────
    print("\n⚙️  Configuring GRPO trainer...")
    grpo_config = GRPOConfig(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_generations=args.num_generations,    # G completions per prompt
        max_completion_length=1024,              # Max tokens for completions
        max_prompt_length=512,                   # Max tokens for prompts
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        # GRPO-specific
        loss_type="dapo",                        # DAPO loss — no length bias
    )

    # ─── Create trainer with reward functions ────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format_correctness,    # Valid tool-call format
            reward_tool_relevance,        # Right tool for the job
            reward_response_quality,      # Empathy, completeness
            reward_thinking_quality,      # Think-plan-act-reflect
        ],
        peft_config=lora_config,
    )

    # ─── Train ───────────────────────────────────────────────────────────
    print("\n🚀 Starting GRPO training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Generations per prompt: {args.num_generations}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Reward functions: format, relevance, quality, thinking")
    print()

    trainer.train()

    # ─── Save ────────────────────────────────────────────────────────────
    print(f"\n💾 Saving GRPO model to {args.output}...")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print("\n✅ GRPO training complete!")
    print(f"   Model saved to: {args.output}")
    print(f"\n   Next step: python scripts/04_quantize_deploy.py --model {args.output}")


if __name__ == "__main__":
    main()
