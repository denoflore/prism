#!/usr/bin/env python3
"""
NIAH fast benchmark — batched + generate-free.
Only checks if needle block lands in MRR top-k (no answer generation).
Batches multiple needle positions per forward pass.
"""
import json, os, time, gc
import torch
import numpy as np

LOG = r"C:\Users\admin\niah_fast_log.txt"
RESULTS_FILE = r"C:\Users\admin\results\niah_50pos.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

MODEL = "Qwen/Qwen2.5-7B"
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
N_POSITIONS = 50
B = 128; K = 32; D_SIG = 32; QUANT_BITS = 5; DRIFT_SIGMA = 0.01

NEEDLE = "The special magic number is 7392158."
FILLER = (
    "This is a passage of filler text used to pad the context window. "
    "It contains no important information and serves only to increase "
    "the total token count. Modern technology advances rapidly across "
    "multiple disciplines including physics and engineering. "
)

# Batch sizes per context length (fit in 49GB VRAM)
BATCH_SIZES = {
    4096: 8,
    8192: 6,
    16384: 3,
    32768: 2,
    65536: 1,
    131072: 1,
}


def build_tokens(tokenizer, target_len, needle_pos_frac):
    """Build token IDs with needle at fractional position. Returns (ids, needle_token_pos)."""
    needle_ids = tokenizer.encode(NEEDLE, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    n_needed = target_len - len(needle_ids) - 10
    all_filler = (filler_ids * ((n_needed // len(filler_ids)) + 2))[:n_needed]
    idx = int(len(all_filler) * needle_pos_frac)
    full = all_filler[:idx] + needle_ids + all_filler[idx:]
    return full[:target_len], idx


def mrr_block_select(hidden, bs, d, k, bits, drift):
    """MRR-impaired block selection. Returns (selected_indices, n_blocks)."""
    seq_len = hidden.shape[0]
    nb = seq_len // bs
    if nb <= k:
        return list(range(nb)), nb
    blocked = hidden[:nb * bs].reshape(nb, bs, -1)
    sigs = blocked.mean(1)[:, :d]
    q = hidden[-1, :d]
    sm = sigs.abs().max() + 1e-10
    W = sigs / sm
    q = q / (q.abs().max() + 1e-10)
    nl = 2 ** bits
    W = torch.round((W + 1) / 2 * (nl - 1)) / (nl - 1) * 2 - 1
    W = W + torch.randn_like(W) * drift
    return torch.topk(W @ q, min(k, nb)).indices.tolist(), nb


def chunked_hidden(model, input_ids, chunk_size=2048):
    """Get last-layer hidden states via chunked forward. Returns CPU tensor."""
    total = input_ids.shape[1]
    past = None
    all_h = []
    for s in range(0, total, chunk_size):
        e = min(s + chunk_size, total)
        out = model(
            input_ids[:, s:e],
            past_key_values=past,
            output_hidden_states=True,
            use_cache=True,
        )
        # Move to CPU immediately to save VRAM
        all_h.append(out.hidden_states[-1][0].cpu())
        past = out.past_key_values
        del out
    # Clear KV cache
    del past
    torch.cuda.empty_cache()
    return torch.cat(all_h, 0)


def process_batch(model, tokenizer, ctx_len, positions, device):
    """Process a batch of needle positions. Returns list of (needle_found, needle_block, n_blocks)."""
    batch_size = len(positions)

    # Build all sequences
    all_ids = []
    needle_positions = []
    for pos in positions:
        ids, npos = build_tokens(tokenizer, ctx_len, pos)
        all_ids.append(ids)
        needle_positions.append(npos)

    results = []

    if batch_size == 1:
        # Single sequence — use chunked forward
        input_ids = torch.tensor([all_ids[0]], dtype=torch.long, device=device)
        hidden = chunked_hidden(model, input_ids).to(device)
        del input_ids
        torch.cuda.empty_cache()

        sel, nb = mrr_block_select(hidden, B, D_SIG, K, QUANT_BITS, DRIFT_SIGMA)
        nblk = needle_positions[0] // B
        results.append((nblk in sel, nblk, nb))
        del hidden
        torch.cuda.empty_cache()
    else:
        # Batch forward — all same length, no padding needed
        input_tensor = torch.tensor(all_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_tensor, output_hidden_states=True)
            hiddens = out.hidden_states[-1]  # (batch, seq, d_model)
            del out
            torch.cuda.empty_cache()

            for i in range(batch_size):
                h = hiddens[i]  # (seq, d_model)
                sel, nb = mrr_block_select(h, B, D_SIG, K, QUANT_BITS, DRIFT_SIGMA)
                nblk = needle_positions[i] // B
                results.append((nblk in sel, nblk, nb))

            del hiddens, input_tensor
            torch.cuda.empty_cache()

    return results


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(LOG, "w") as f:
        f.write("")

    log("=" * 60)
    log(f"NIAH FAST: {N_POSITIONS} positions x {len(CONTEXT_LENGTHS)} contexts")
    log(f"Batched + generate-free (block selection only)")
    log(f"K={K}, B={B}, d_sig={D_SIG}, {QUANT_BITS}-bit MRR")
    log("=" * 60)

    device = "cuda"
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dk = "dtype" if tv >= (4, 57) else "torch_dtype"

    log(f"GPU: {torch.cuda.get_device_name()}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, **{dk: torch.bfloat16},
        device_map="cuda", trust_remote_code=True
    )
    model.eval()
    log(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    positions = np.linspace(0.05, 0.95, N_POSITIONS).tolist()
    all_results = {}

    for ctx_len in CONTEXT_LENGTHS:
        batch_size = BATCH_SIZES.get(ctx_len, 1)
        log(f"\n{'=' * 40}")
        log(f"Context: {ctx_len} ({ctx_len // 1024}K) | batch={batch_size}")
        log(f"{'=' * 40}")

        ctx_results = []
        t0 = time.time()
        n_batches = (N_POSITIONS + batch_size - 1) // batch_size

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, N_POSITIONS)
            batch_pos = positions[start:end]

            try:
                with torch.no_grad():
                    batch_results = process_batch(model, tokenizer, ctx_len, batch_pos, device)

                for j, (found, nblk, nb) in enumerate(batch_results):
                    idx = start + j
                    ctx_results.append({"needle_found": found, "needle_block": nblk, "n_blocks": nb})
                    log(f"  [{idx + 1}/{N_POSITIONS}] pos={positions[idx]:.2f} "
                        f"blk={nblk} found={found}")

            except torch.cuda.OutOfMemoryError:
                log(f"  Batch {bi + 1}/{n_batches} OOM — falling back to single")
                torch.cuda.empty_cache()
                gc.collect()
                # Fallback: process one by one
                for j, pos in enumerate(batch_pos):
                    idx = start + j
                    try:
                        res = process_batch(model, tokenizer, ctx_len, [pos], device)
                        found, nblk, nb = res[0]
                        ctx_results.append({"needle_found": found, "needle_block": nblk, "n_blocks": nb})
                        log(f"  [{idx + 1}/{N_POSITIONS}] pos={pos:.2f} blk={nblk} found={found}")
                    except torch.cuda.OutOfMemoryError:
                        log(f"  [{idx + 1}/{N_POSITIONS}] OOM skip")
                        torch.cuda.empty_cache()
                        gc.collect()
                        ctx_results.append({"needle_found": False, "error": "OOM"})
                    except Exception as e:
                        log(f"  [{idx + 1}/{N_POSITIONS}] ERROR: {e}")
                        ctx_results.append({"needle_found": False, "error": str(e)})

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        n_valid = sum(1 for r in ctx_results if "error" not in r)
        needle_recall = sum(1 for r in ctx_results if r.get("needle_found")) / max(n_valid, 1)

        all_results[str(ctx_len)] = {
            "needle_recall": needle_recall,
            "n_valid": n_valid,
            "n_positions": N_POSITIONS,
            "elapsed_s": elapsed,
        }
        log(f"  >> Needle recall: {needle_recall:.1%} | Valid: {n_valid}/{N_POSITIONS} | Time: {elapsed:.0f}s")
        log(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "model": MODEL,
            "mode": "fast_batch_selection_only",
            "parameters": {
                "n_positions": N_POSITIONS,
                "context_lengths": CONTEXT_LENGTHS,
                "batch_sizes": BATCH_SIZES,
                "B": B, "K": K, "d_sig": D_SIG,
                "quant_bits": QUANT_BITS, "drift_sigma": DRIFT_SIGMA,
            },
            "results": all_results,
        }, f, indent=2)

    log(f"\nSaved: {RESULTS_FILE}")
    log("Done!")


if __name__ == "__main__":
    main()
