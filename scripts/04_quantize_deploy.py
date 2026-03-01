#!/usr/bin/env python3
"""
04_quantize_deploy.py — Merge LoRA adapters, convert to GGUF, quantize to Q4_K_M,
and deploy to Orin Nano via Ollama.

Runs on RTX 5090 for merging/conversion, then deploys to Orin Nano.

Prerequisites:
    pip install torch transformers peft accelerate
    # For GGUF conversion, need llama.cpp:
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make

Usage:
    # Step 1: Merge LoRA + Convert to GGUF (on 5090)
    python scripts/04_quantize_deploy.py --model models/ministral-3b-grpo --output models/ministral-3b-gguf

    # Step 2: Deploy to Orin (see printed instructions)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(model_path: str, output_path: str):
    """Merge LoRA adapters back into the base model."""
    print("🔀 Merging LoRA adapters into base model...")

    # Find base model — check adapter_config.json first, then training_info.json
    adapter_cfg_path = Path(model_path) / "adapter_config.json"
    info_path = Path(model_path) / "training_info.json"
    base_model = None

    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path")

    if not base_model and info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        base_model = info.get("base_model")

    if not base_model:
        base_model = os.getenv("BASE_MODEL", "mistralai/Ministral-3-3B-Instruct-2512")

    print(f"   Base model: {base_model}")

    # Detect model architecture
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    # Load base model in full precision for merging
    print("   Loading base model (this may take a minute)...")
    if model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        # Check for FP8 quantization
        existing_quant = getattr(config, "quantization_config", None)
        if existing_quant and isinstance(existing_quant, dict) and existing_quant.get("quant_method"):
            from transformers import FineGrainedFP8Config
            print(f"   📎 Base model is FP8 — dequantizing to BF16 for merge...")
            model = Mistral3ForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=FineGrainedFP8Config(dequantize=True),
            )
            if hasattr(model.config, 'quantization_config'):
                model.config.quantization_config = None
            if hasattr(model, 'is_quantized'):
                model.is_quantized = False
            if hasattr(model, 'hf_quantizer'):
                model.hf_quantizer = None
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load LoRA adapter
    print(f"   Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)

    # Merge and unload
    print("   Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    merged_path = Path(output_path) / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"   Saving merged model to {merged_path}...")

    # Ensure all quantization artifacts are fully cleared before saving
    # (FP8 dequantized models can fail on save_pretrained otherwise)
    if hasattr(model, 'hf_quantizer'):
        model.hf_quantizer = None
    if hasattr(model, 'is_quantized'):
        model.is_quantized = False
    if hasattr(model.config, 'quantization_config'):
        model.config.quantization_config = None
    # Remove _pre_quantization_dtype if present (causes issues with some save paths)
    if hasattr(model, '_pre_quantization_dtype'):
        delattr(model, '_pre_quantization_dtype')

    # Clear weight conversion metadata left by FP8 dequantization
    # (these contain FP8 ops that can't be reversed, causing NotImplementedError on save)
    if hasattr(model, '_weight_conversions'):
        model._weight_conversions = None

    try:
        model.save_pretrained(str(merged_path), safe_serialization=True)
    except (NotImplementedError, Exception) as e:
        print(f"   ⚠️  save_pretrained failed ({e}), trying state_dict save...")
        import gc
        from safetensors.torch import save_file

        # Manual save: get state dict, save as safetensors + config
        state_dict = model.state_dict()
        # Move all tensors to CPU
        state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
        gc.collect()
        torch.cuda.empty_cache()

        # Save weights
        save_file(state_dict, str(merged_path / "model.safetensors"))

        # Save config
        model.config.save_pretrained(str(merged_path))

        print("   ✅ Saved via manual state_dict export")

    tokenizer.save_pretrained(str(merged_path))

    print("   ✅ LoRA merge complete!")
    return str(merged_path)


def convert_to_gguf(merged_path: str, output_path: str, llama_cpp_path: str):
    """Convert HuggingFace model to GGUF format using llama.cpp."""
    print("\n📦 Converting to GGUF format...")

    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"   ❌ llama.cpp convert script not found at {convert_script}")
        print(f"   Please clone llama.cpp: git clone https://github.com/ggml-org/llama.cpp")
        sys.exit(1)

    gguf_output = Path(output_path) / "model-f16.gguf"
    cmd = [
        sys.executable, str(convert_script),
        merged_path,
        "--outfile", str(gguf_output),
        "--outtype", "f16",
    ]

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   ❌ Conversion failed: {result.stderr}")
        sys.exit(1)

    print(f"   ✅ GGUF file created: {gguf_output}")
    return str(gguf_output)


def quantize_gguf(gguf_path: str, output_path: str, llama_cpp_path: str, quant_type: str = "Q4_K_M"):
    """Quantize GGUF model to specified quantization level."""
    print(f"\n🗜️  Quantizing to {quant_type}...")

    # Search multiple possible locations for llama-quantize
    search_paths = [
        Path(llama_cpp_path) / "llama-quantize",
        Path(llama_cpp_path) / "build" / "bin" / "llama-quantize",
        Path(llama_cpp_path) / "build_cpu" / "bin" / "llama-quantize",
        Path(llama_cpp_path) / "bin" / "llama-quantize",
    ]
    quantize_bin = None
    for p in search_paths:
        if p.exists():
            quantize_bin = p
            break
    if quantize_bin is None:
        print(f"   ❌ llama-quantize not found. Build llama.cpp first:")
        print(f"      cd {llama_cpp_path} && cmake -B build_cpu -DGGML_CUDA=OFF && cmake --build build_cpu --target llama-quantize -j$(nproc)")
        sys.exit(1)

    output_file = Path(output_path) / f"model-{quant_type.lower()}.gguf"
    cmd = [str(quantize_bin), gguf_path, str(output_file), quant_type]

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        print(f"   ❌ Quantization failed: {stderr_text}")
        sys.exit(1)

    # Get file size
    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"   ✅ Quantized model: {output_file} ({size_mb:.1f} MB)")

    return str(output_file)


def create_ollama_modelfile(gguf_path: str, output_path: str):
    """Create an Ollama Modelfile for easy deployment."""
    print("\n📝 Creating Ollama Modelfile...")

    modelfile_content = f"""# Reachy Copilot — Ministral fine-tuned for embodied AI
FROM {gguf_path}

# System prompt baked into the model
SYSTEM \"\"\"You are Reachy, an embodied AI assistant running on a Reachy Mini robot. You can:
- Search the web for real-time information
- Control your robotic head (look at things, express emotions)
- Speak aloud to the user
- Access patient health records (with permission)
- Set reminders

When responding:
1. THINK about the user's intent in <think></think> tags
2. PLAN which tools to call
3. ACT by calling tools with <tool_call></tool_call> tags
4. REFLECT on whether your response fully addressed the user's needs

Be empathetic, professional, and proactive.\"\"\"

# Parameters tuned for Orin Nano (8GB)
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER num_predict 512

# Template for Mistral chat format
TEMPLATE \"\"\"{{{{if .System}}}}<|im_start|>system
{{{{.System}}}}<|im_end|>
{{{{end}}}}{{{{if .Prompt}}}}<|im_start|>user
{{{{.Prompt}}}}<|im_end|>
{{{{end}}}}<|im_start|>assistant
{{{{.Response}}}}<|im_end|>\"\"\"
"""

    modelfile_path = Path(output_path) / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"   ✅ Modelfile created: {modelfile_path}")
    return str(modelfile_path)


def print_deployment_instructions(output_path: str, quant_type: str):
    """Print step-by-step deployment instructions for Orin Nano."""
    gguf_name = f"model-{quant_type.lower()}.gguf"
    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT INSTRUCTIONS — Orin Nano")
    print("=" * 60)
    print(f"""
Your quantized model is ready at:
  {output_path}/{gguf_name}

━━━ Option A: Ollama (Recommended) ━━━━━━━━━━━━━━━━━━━━━━━━

1. Copy files to Orin Nano:
   scp -r {output_path}/ orin@<ORIN_IP>:~/reachy-model/

2. On the Orin Nano, install Ollama (if not already):
   curl -fsSL https://ollama.com/install.sh | sh

3. Create the model in Ollama:
   cd ~/reachy-model/
   ollama create reachy-copilot -f Modelfile

4. Test the model:
   ollama run reachy-copilot "Hello, I'm your patient. How can you help?"

5. Serve via API (for OpenClaw integration):
   # Ollama serves on port 11434 by default
   curl http://localhost:11434/api/generate -d '{{
     "model": "reachy-copilot",
     "prompt": "Search for flu symptoms and tell me about them"
   }}'

━━━ Option B: llama.cpp server ━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Copy the GGUF to Orin:
   scp {output_path}/{gguf_name} orin@<ORIN_IP>:~/

2. On the Orin, build llama.cpp with CUDA:
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j$(nproc)

3. Run the server:
   ./build/bin/llama-server \\
     -m ~/{gguf_name} \\
     -c 2048 \\
     -ngl 99 \\
     --host 0.0.0.0 \\
     --port 8080

4. Test:
   curl http://localhost:8080/v1/chat/completions -d '{{
     "messages": [{{"role": "user", "content": "Hello!"}}]
   }}'

━━━ Memory Budget (Orin Nano 8GB) ━━━━━━━━━━━━━━━━━━━━━━━━
  Model (Q4_K_M ~2GB)   : ~2.0 GB
  KV Cache (2048 ctx)    : ~0.5 GB
  CUDA runtime           : ~0.8 GB
  OS + Reachy SDK        : ~1.5 GB
  ────────────────────────────────
  Total                  : ~4.8 GB ✅ (3.2 GB headroom)

━━━ Verify on Orin ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Check CUDA is available
  python3 -c "import torch; print(torch.cuda.is_available())"

  # Check Ollama status
  ollama list
  ollama ps

  # Check memory usage
  tegrastats
""")


def main():
    parser = argparse.ArgumentParser(description="Quantize and deploy model to Orin Nano")
    parser.add_argument("--model", type=str, required=True, help="Path to SFT/GRPO checkpoint")
    parser.add_argument("--output", type=str, default="models/ministral-3b-gguf", help="Output directory")
    parser.add_argument("--llama-cpp", type=str, default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--quant", type=str, default="Q4_K_M", help="Quantization type (Q4_K_M, Q4_K_S, Q5_K_M, Q8_0)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge (model is already merged)")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--gguf-path", type=str, help="Path to existing GGUF file (skip merge + convert)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📦 Quantize & Deploy — Ministral → Orin Nano")
    print("=" * 60)

    if args.gguf_path:
        # Skip everything, just quantize
        gguf_path = args.gguf_path
    else:
        # Step 1: Merge LoRA
        if not args.skip_merge:
            merged_path = merge_lora(args.model, str(output_path))
        else:
            merged_path = args.model
            print(f"\n⏭️  Skipping merge, using: {merged_path}")

        # Step 2: Convert to GGUF
        if not args.skip_convert:
            gguf_path = convert_to_gguf(merged_path, str(output_path), args.llama_cpp)
        else:
            gguf_path = str(output_path / "model-f16.gguf")
            print(f"\n⏭️  Skipping convert, using: {gguf_path}")

    # Step 3: Quantize
    quantized_path = quantize_gguf(gguf_path, str(output_path), args.llama_cpp, args.quant)

    # Step 4: Create Ollama Modelfile
    create_ollama_modelfile(os.path.basename(quantized_path), str(output_path))

    # Step 5: Print deployment instructions
    print_deployment_instructions(str(output_path), args.quant)

    print("\n✅ Quantization complete!")
    print(f"   Output directory: {args.output}")


if __name__ == "__main__":
    main()
