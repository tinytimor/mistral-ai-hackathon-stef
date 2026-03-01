#!/usr/bin/env python3
"""
07_download_models.py - Download and cache-check all models for the pipeline.

This script ensures all required models are downloaded BEFORE training or
deployment begins. It supports three modes:

  1. Training mode (default): Downloads base HuggingFace models needed for SFT/GRPO
  2. No-train mode (--no-train): Downloads pre-quantized GGUFs for direct edge deployment
  3. Edge mode (--edge-stack): Downloads the full Orin Nano inference stack
     (Ministral 3B GGUF + Voxtral Mini 3B GGUF + Whisper tiny)

Voxtral Mini 3B on Orin Nano (8GB):
  - Q4_K_M = 2.47 GB - fits alongside Ministral 3B Q4 (2.0 GB)
  - BUT they can't run simultaneously. Use model-swap strategy:
    * Ministral 3B for text tool-calling (always loaded)
    * Voxtral Mini 3B for audio transcription + understanding (loaded on demand)
  - Voxtral does ASR + audio Q&A + function calling FROM VOICE natively
  - Replaces separate Whisper STT → reduces pipeline complexity

Memory budget with Voxtral swap strategy (Orin Nano 8GB):
  Ministral 3B Q4_K_M        : ~2.0 GB (primary, always loaded)
  KV Cache (2048 ctx)         : ~0.5 GB
  CUDA runtime                : ~0.8 GB
  OS + Reachy SDK + bridge    : ~1.5 GB
  Piper TTS                   : ~0.2 GB
  ─────────────────────────────────────
  Total (text mode)           : ~5.0 GB  ✅ (3.0 GB headroom)

  Voxtral Mini 3B Q4_K_M     : ~2.5 GB (swapped in for audio tasks)
  - Swap out Ministral, load Voxtral for STT + audio understanding
  - Or run both at Q3_K quantization (~1.8 GB each = 3.6 GB combined)

Usage:
    # Check & download models for training pipeline:
    python scripts/07_download_models.py

    # Download pre-quantized GGUFs (skip training entirely):
    python scripts/07_download_models.py --no-train

    # Download full edge inference stack for Orin Nano:
    python scripts/07_download_models.py --edge-stack

    # Just check what's cached (dry run):
    python scripts/07_download_models.py --check-only

    # Download to a specific directory:
    python scripts/07_download_models.py --no-train --output ./models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ─── Model Definitions ───────────────────────────────────────────────────────

# Base models for training (HuggingFace format, need HF_TOKEN for gated models)
TRAINING_MODELS = {
    "ministral-3b": {
        "repo_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "description": "Ministral 3 3B - primary student model for SFT/GRPO",
        "size_gb": 6.9,
        "gated": False,
    },
    "ministral-8b": {
        "repo_id": "mistralai/Ministral-3-8B-Instruct-2512",
        "description": "Ministral 3 8B - larger student model (if VRAM allows)",
        "size_gb": 16.0,
        "gated": False,
    },
}

# Pre-quantized GGUFs for direct deployment (no training needed)
GGUF_MODELS = {
    "ministral-3b-q4": {
        "repo_id": "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        "filename": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        "description": "Ministral 3 3B Q4_K_M - tool-calling on Orin Nano",
        "size_gb": 2.15,
        "gated": False,
    },
    "ministral-3b-q5": {
        "repo_id": "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        "filename": "Ministral-3-3B-Instruct-2512-Q5_K_M.gguf",
        "description": "Ministral 3 3B Q5_K_M - higher quality, still fits Orin",
        "size_gb": 2.47,
        "gated": False,
    },
}

# Edge inference stack (all models needed on Orin Nano)
EDGE_MODELS = {
    "voxtral-mini-3b-q4": {
        "repo_id": "mradermacher/Voxtral-Mini-3B-2507-GGUF",
        "filename": "Voxtral-Mini-3B-2507.Q4_K_M.gguf",
        "description": "Voxtral Mini 3B Q4_K_M - voice STT + audio understanding + function calling",
        "size_gb": 2.5,
        "gated": False,
        "note": "Replaces Whisper for STT - does transcription, Q&A, and tool-calling from audio natively",
    },
    "voxtral-mini-3b-q5": {
        "repo_id": "mradermacher/Voxtral-Mini-3B-2507-GGUF",
        "filename": "Voxtral-Mini-3B-2507.Q5_K_M.gguf",
        "description": "Voxtral Mini 3B Q5_K_M - higher quality audio model",
        "size_gb": 2.9,
        "gated": False,
    },
}


def check_hf_cache(repo_id: str, filename: str = None) -> tuple[bool, str]:
    """Check if a model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir, hf_hub_download
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                if filename:
                    # Check for specific file
                    for revision in repo.revisions:
                        for f in revision.files:
                            if f.file_name == filename:
                                return True, str(f.file_path)
                else:
                    # Any revision with files counts
                    for revision in repo.revisions:
                        if revision.files:
                            return True, str(revision.snapshot_path)
    except Exception:
        pass
    return False, ""


def check_local_file(output_dir: str, filename: str) -> tuple[bool, str]:
    """Check if a model file exists locally."""
    path = Path(output_dir) / filename
    if path.exists():
        return True, str(path)
    return False, ""


def download_hf_model(repo_id: str, output_dir: str = None,
                       filename: str = None, token: str = None) -> str:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    kwargs = {}
    if token:
        kwargs["token"] = token

    if filename:
        # Download single GGUF file
        local_dir = output_dir or "models"
        print(f"   Downloading {filename} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            **kwargs,
        )
        return path
    else:
        # Download full model (for training)
        print(f"   Downloading full model {repo_id}...")
        path = snapshot_download(
            repo_id=repo_id,
            **kwargs,
        )
        return path


def create_ollama_modelfile(gguf_path: str, output_dir: str, model_name: str = "reachy-copilot"):
    """Create an Ollama Modelfile for edge deployment."""
    modelfile = f"""# Reachy Copilot - {model_name}
# Generated by 07_download_models.py for edge deployment on Orin Nano
FROM {gguf_path}

SYSTEM \"\"\"You are Reachy, an embodied AI assistant running on a Reachy Mini robot.
You can control your robotic body, search the web, and help your user.

When you need to take action, use tool calls:
<tool_call>{{"name": "tool_name", "arguments": {{}}}}</tool_call>

Available tools:
- robot_look_at: Move head to look at a point (x, y, z)
- robot_express: Show emotion (happy, sad, curious, surprised, thinking)
- robot_speak: Speak text aloud through the robot speaker
- robot_nod: Quick nod gesture
- search_web: Search the internet (query)
- set_reminder: Set a timed reminder (message, minutes)
- memory_search: Search conversation memory (query)

Be concise, warm, and helpful. Express emotions physically when appropriate.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER num_predict 512

TEMPLATE \"\"\"{{{{if .System}}}}<s>[INST] {{{{.System}}}}

{{{{end}}}}{{{{if .Prompt}}}}{{{{.Prompt}}}} [/INST]{{{{end}}}}{{{{if .Response}}}} {{{{.Response}}}}</s>{{{{end}}}}\"\"\"
"""
    modelfile_path = Path(output_dir) / "Modelfile"
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    print(f"   ✅ Ollama Modelfile created: {modelfile_path}")
    return str(modelfile_path)


def print_model_status(models: dict, mode: str, output_dir: str, token: str = None):
    """Check and display status of all required models."""
    print(f"\n{'='*60}")
    print(f"  📦 Model Status ({mode})")
    print(f"{'='*60}\n")

    all_cached = True
    for key, info in models.items():
        filename = info.get("filename")
        repo_id = info["repo_id"]

        # Check local output dir first
        if filename:
            cached, path = check_local_file(output_dir, filename)
            if not cached:
                cached, path = check_hf_cache(repo_id, filename)
        else:
            cached, path = check_hf_cache(repo_id)

        status = "✅ cached" if cached else "⬇️  needs download"
        gated = " 🔒 (needs HF_TOKEN)" if info.get("gated") and not cached else ""
        size = f"~{info['size_gb']:.1f} GB"

        print(f"  {status} {info['description']}")
        print(f"         {repo_id}{gated} ({size})")
        if cached:
            print(f"         → {path}")
        print()

        if not cached:
            all_cached = False

    return all_cached


def main():
    parser = argparse.ArgumentParser(description="Download and cache-check Mistral models")
    parser.add_argument("--no-train", action="store_true",
                        help="Download pre-quantized GGUFs for direct deployment (skip training)")
    parser.add_argument("--edge-stack", action="store_true",
                        help="Download full edge stack (Ministral + Voxtral GGUFs)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check cache status, don't download")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory for downloaded models")
    parser.add_argument("--base-model", type=str, default="ministral-3b",
                        choices=list(TRAINING_MODELS.keys()),
                        help="Which base model to download for training")
    parser.add_argument("--include-voxtral", action="store_true",
                        help="Also download Voxtral Mini 3B GGUF for audio understanding on edge")
    args = parser.parse_args()

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    print("=" * 60)
    print("  📦 Reachy Copilot - Model Download & Cache Check")
    print("=" * 60)

    if args.no_train or args.edge_stack:
        # ─── No-train / Edge mode: download pre-quantized GGUFs ──────────
        models_to_get = dict(GGUF_MODELS)
        mode = "Pre-quantized GGUFs (no training)"

        if args.edge_stack or args.include_voxtral:
            models_to_get.update(EDGE_MODELS)
            mode = "Full Edge Stack (Ministral + Voxtral GGUFs)"

        all_cached = print_model_status(models_to_get, mode, output_dir, token)

        if args.check_only:
            if all_cached:
                print("✅ All models cached and ready!")
            else:
                print("⚠️  Some models need downloading. Run without --check-only to download.")
            return

        if all_cached:
            print("✅ All models already cached!")
        else:
            print("\n⬇️  Downloading missing models...\n")
            for key, info in models_to_get.items():
                filename = info.get("filename")
                cached, _ = check_local_file(output_dir, filename)
                if not cached:
                    cached, _ = check_hf_cache(info["repo_id"], filename)
                if cached:
                    continue

                print(f"  📥 {info['description']}")
                try:
                    path = download_hf_model(
                        info["repo_id"], output_dir, filename, token
                    )
                    print(f"     ✅ Downloaded → {path}")
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
                    if info.get("gated"):
                        print("     💡 This model may require authentication.")
                        print("        Set HF_TOKEN in your .env file.")
                        print("        Get a token at: https://huggingface.co/settings/tokens")

        # Create Modelfile for the primary GGUF
        primary_gguf = GGUF_MODELS["ministral-3b-q4"]["filename"]
        primary_path = Path(output_dir) / primary_gguf
        if primary_path.exists():
            create_ollama_modelfile(primary_gguf, output_dir)

        # Print deployment commands
        print(f"\n{'='*60}")
        print(f"  🚀 Quick Deploy to Orin Nano")
        print(f"{'='*60}")
        print(f"""
  1. Copy models to Orin Nano:
     scp -r {output_dir}/ orin@<ORIN_IP>:~/reachy-model/

  2. On the Orin, create Ollama model:
     cd ~/reachy-model && ollama create reachy-copilot -f Modelfile

  3. Test it:
     ollama run reachy-copilot "Hello! What can you do?"

  4. Start the bridge server:
     python scripts/06_openclaw_bridge.py --standalone --reachy-ip localhost
""")

        if args.edge_stack or args.include_voxtral:
            print(f"""  🎤 Voxtral Audio (optional - swap in for audio tasks):
     # Voxtral replaces Whisper for STT - does transcription + understanding + tool-calling from audio
     # Create a separate Ollama model for audio mode:
     # (manually create Modelfile.voxtral pointing to Voxtral GGUF)
     ollama create reachy-voxtral -f Modelfile.voxtral
     
  📊 Orin Nano 8GB Memory Budget:
     Text mode:  Ministral 3B Q4 (2.0) + KV cache (0.5) + CUDA (0.8) + OS (1.5) = 4.8 GB ✅
     Audio mode: Voxtral Mini 3B Q4 (2.5) + KV cache (0.5) + CUDA (0.8) + OS (1.5) = 5.3 GB ✅
     Both (Q3):  Both at Q3_K (~1.8 + 1.8) + KV (0.5) + CUDA (0.8) + OS (1.5)     = 6.4 GB ✅
""")

    else:
        # ─── Training mode: download full HuggingFace models ─────────────
        model_key = args.base_model
        model_info = TRAINING_MODELS[model_key]
        models_to_check = {model_key: model_info}

        mode = "Training (HuggingFace format)"
        all_cached = print_model_status(models_to_check, mode, output_dir, token)

        if args.check_only:
            if all_cached:
                print("✅ Base model cached and ready for training!")
            else:
                print("⚠️  Base model needs downloading. Run without --check-only.")
                if model_info.get("gated"):
                    print(f"\n💡 {model_info['repo_id']} is a gated model.")
                    print("   1. Accept the license at: https://huggingface.co/" + model_info["repo_id"])
                    print("   2. Set HF_TOKEN in your .env file")
            return

        if not all_cached:
            if model_info.get("gated") and not token:
                print(f"\n❌ {model_info['repo_id']} is a gated model and HF_TOKEN is not set!")
                print("   1. Go to https://huggingface.co/" + model_info["repo_id"])
                print("   2. Accept the license agreement")
                print("   3. Create a token at https://huggingface.co/settings/tokens")
                print("   4. Set HF_TOKEN in your .env file")
                sys.exit(1)

            print(f"\n⬇️  Downloading {model_info['description']}...")
            print(f"   This may take a while (~{model_info['size_gb']:.0f} GB)...\n")
            try:
                path = download_hf_model(model_info["repo_id"], token=token)
                print(f"\n✅ Model downloaded and cached!")
                print(f"   Path: {path}")
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
                if model_info.get("gated"):
                    print("   💡 Make sure you've accepted the model license on HuggingFace")
                sys.exit(1)
        else:
            print("✅ Base model already cached!")

        # Also check if user wants Voxtral for edge
        if args.include_voxtral:
            print("\n📥 Also downloading Voxtral for edge audio understanding...")
            voxtral = EDGE_MODELS["voxtral-mini-3b-q4"]
            cached, _ = check_local_file(output_dir, voxtral["filename"])
            if not cached:
                cached, _ = check_hf_cache(voxtral["repo_id"], voxtral["filename"])
            if not cached:
                try:
                    download_hf_model(voxtral["repo_id"], output_dir, voxtral["filename"])
                    print("   ✅ Voxtral Mini 3B GGUF downloaded!")
                except Exception as e:
                    print(f"   ⚠️  Voxtral download failed: {e} (non-fatal)")

    print("\n✅ Model download/check complete!")


if __name__ == "__main__":
    main()
