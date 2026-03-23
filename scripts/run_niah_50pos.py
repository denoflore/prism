#!/usr/bin/env python3
"""
NIAH 50-position benchmark — memory-efficient version.
Processes one context length at a time, clears GPU memory between runs.
Uses chunked prefill for long contexts to avoid OOM.
"""
import json, os, sys, time, gc
import torch
import numpy as np

LOG = r"C:\Users\admin\niah_50pos_log.txt"
RESULTS_FILE = r"C:\Users\admin\results\niah_50pos.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Parameters ──
MODEL = "Qwen/Qwen2.5-7B"
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536]
N_POSITIONS = 50
B = 128
K = 32
D_SIG = 32
QUANT_BITS = 5
DRIFT_SIGMA = 0.01

NEEDLE = "The special magic number is 7392158."
QUESTION = "What is the special magic number mentioned in the text?"
EXPECTED = "7392158"

FILLER = (
    "This is a passage of filler text used to pad the context window. "
    "It contains no important information and serves only to increase "
    "the total token count. Modern technology advances rapidly across "
    "multiple disciplines including physics and engineering. "
)


def build_context(tokenizer, target_len, needle_pos_frac):
    """Build tokenized context with needle at given fractional position."""
    needle_ids = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_ids = tokenizer.encode(f"\n\nQuestion: {QUESTION}\nAnswer:", add_special_tokens=False)

    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    n_filler_needed = target_len - len(needle_ids) - len(question_ids) - 10
    n_repeats = (n_filler_needed // len(filler_ids)) + 2
    all_filler = (filler_ids * n_repeats)[:n_filler_needed]

    insert_idx = int(len(all_filler) * needle_pos_frac)
    context_ids = all_filler[:insert_idx] + needle_ids + all_filler[insert_idx:]

    full_ids = context_ids + question_ids
    return full_ids[:target_len], insert_idx


def mrr_block_select(hidden_states, block_size, d_sig, k, bits, drift_sigma):
    """Simulate MRR block selection on hidden states. Returns selected block indices."""
    seq_len, d_model = hidden_states.shape
    n_blocks = seq_len // block_size
    if n_blocks <= k:
        return list(range(n_blocks)), n_blocks

    # Reshape into blocks and compute mean-key signatures
    blocked = hidden_states[:n_blocks * block_size].reshape(n_blocks, block_size, d_model)
    sigs = blocked.mean(dim=1)[:, :d_sig]  # (n_blocks, d_sig)
    query = hidden_states[-1, :d_sig]  # last token as query

    # Normalize for balanced PD
    s_max = sigs.abs().max() + 1e-10
    W = sigs / s_max
    q = query / (query.abs().max() + 1e-10)

    # Quantize
    n_levels = 2 ** bits
    W_q = torch.round((W + 1) / 2 * (n_levels - 1)) / (n_levels - 1) * 2 - 1

    # Thermal drift
    W_q = W_q + torch.randn_like(W_q) * drift_sigma

    # Inner products + top-k
    scores = W_q @ q
    topk = torch.topk(scores, min(k, n_blocks)).indices.tolist()
    return topk, n_blocks


def run_single(model, tokenizer, ctx_len, needle_pos_frac, device):
    """Run one NIAH trial with memory-efficient approach."""
    input_ids, needle_token_pos = build_context(tokenizer, ctx_len, needle_pos_frac)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        # Chunked forward to avoid OOM — process in chunks of 4096
        chunk_size = 4096
        total_len = input_tensor.shape[1]

        if total_len <= chunk_size:
            outputs = model(input_tensor, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]  # (seq_len, d_model)
        else:
            # Use model's built-in KV cache for chunked prefill
            past_key_values = None
            all_hidden = []

            for start in range(0, total_len, chunk_size):
                end = min(start + chunk_size, total_len)
                chunk = input_tensor[:, start:end]

                outputs = model(
                    chunk,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                all_hidden.append(outputs.hidden_states[-1][0])
                past_key_values = outputs.past_key_values

                del outputs
                torch.cuda.empty_cache()

            hidden = torch.cat(all_hidden, dim=0)
            del all_hidden, past_key_values
            torch.cuda.empty_cache()

        # MRR block selection
        selected_blocks, n_blocks = mrr_block_select(
            hidden, B, D_SIG, K, QUANT_BITS, DRIFT_SIGMA
        )

        needle_block = needle_token_pos // B
        needle_found = needle_block in selected_blocks

        del hidden
        torch.cuda.empty_cache()

        # Generate answer
        gen_out = model.generate(
            input_tensor, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(gen_out[0][total_len:], skip_special_tokens=True).strip()
        correct = EXPECTED in answer

        del input_tensor, gen_out
        torch.cuda.empty_cache()

    return {
        "correct": correct,
        "needle_found": needle_found,
        "answer": answer[:50],
        "n_blocks": n_blocks,
        "needle_block": needle_block,
    }


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(LOG, "w") as f:
        f.write("")

    log("=" * 60)
    log(f"NIAH Benchmark: {N_POSITIONS} positions x {len(CONTEXT_LENGTHS)} contexts")
    log(f"Model: {MODEL} | K={K}, B={B}, d_sig={D_SIG}, {QUANT_BITS}-bit MRR")
    log("Memory-efficient: chunked prefill + per-context cleanup")
    log("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dk = "dtype" if tv >= (4, 57) else "torch_dtype"

    log(f"Device: {device}, transformers {transformers.__version__}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name()}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, **{dk: torch.bfloat16},
        device_map="cuda", trust_remote_code=True
    )
    model.eval()
    log(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    positions = np.linspace(0.05, 0.95, N_POSITIONS)
    results = {}

    for ctx_len in CONTEXT_LENGTHS:
        log(f"\n{'='*40}")
        log(f"Context: {ctx_len} tokens ({ctx_len//1024}K)")
        log(f"{'='*40}")

        ctx_results = []
        t0 = time.time()

        for i, pos in enumerate(positions):
            try:
                r = run_single(model, tokenizer, ctx_len, pos, device)
                ctx_results.append(r)
                status = "OK" if r["correct"] else "MISS"
                log(f"  [{i+1}/{N_POSITIONS}] pos={pos:.2f} {status} "
                    f"needle_blk={r['needle_block']} found={r['needle_found']}")
            except torch.cuda.OutOfMemoryError:
                log(f"  [{i+1}/{N_POSITIONS}] pos={pos:.2f} OOM — skipping")
                torch.cuda.empty_cache()
                gc.collect()
                ctx_results.append({"correct": False, "needle_found": False, "error": "OOM"})
            except Exception as e:
                log(f"  [{i+1}/{N_POSITIONS}] pos={pos:.2f} ERROR: {e}")
                ctx_results.append({"correct": False, "needle_found": False, "error": str(e)})

            # Periodic cleanup
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        n_valid = sum(1 for r in ctx_results if "error" not in r)
        accuracy = sum(1 for r in ctx_results if r.get("correct")) / max(n_valid, 1)
        needle_recall = sum(1 for r in ctx_results if r.get("needle_found")) / max(n_valid, 1)

        results[str(ctx_len)] = {
            "accuracy": accuracy,
            "needle_recall": needle_recall,
            "n_valid": n_valid,
            "n_positions": N_POSITIONS,
            "elapsed_s": elapsed,
        }
        log(f"  >> Accuracy: {accuracy:.1%} | Needle recall: {needle_recall:.1%} "
            f"| Valid: {n_valid}/{N_POSITIONS} | Time: {elapsed:.0f}s")

        # Aggressive cleanup between context lengths
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  VRAM after cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "model": MODEL,
            "parameters": {
                "n_positions": N_POSITIONS,
                "context_lengths": CONTEXT_LENGTHS,
                "B": B, "K": K, "d_sig": D_SIG,
                "quant_bits": QUANT_BITS, "drift_sigma": DRIFT_SIGMA,
            },
            "results": results,
        }, f, indent=2)

    log(f"\nSaved: {RESULTS_FILE}")
    log("Done!")


if __name__ == "__main__":
    main()
