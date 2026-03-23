#!/usr/bin/env python3
"""NIAH 128K only — runs after 50pos finishes."""
import json, os, sys, time, gc
import torch
import numpy as np

LOG = r"C:\Users\admin\niah_128k_log.txt"
RESULTS_FILE = r"C:\Users\admin\results\niah_128k.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

MODEL = "Qwen/Qwen2.5-7B"
CTX = 131072
N_POSITIONS = 50
B = 128; K = 32; D_SIG = 32; QUANT_BITS = 5; DRIFT_SIGMA = 0.01

NEEDLE = "The special magic number is 7392158."
QUESTION = "What is the special magic number mentioned in the text?"
EXPECTED = "7392158"
FILLER = (
    "This is a passage of filler text used to pad the context window. "
    "It contains no important information and serves only to increase "
    "the total token count. Modern technology advances rapidly across "
    "multiple disciplines including physics and engineering. "
)

def build_context(tokenizer, target_len, pos_frac):
    needle_ids = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_ids = tokenizer.encode(f"\n\nQuestion: {QUESTION}\nAnswer:", add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    n_needed = target_len - len(needle_ids) - len(question_ids) - 10
    all_filler = (filler_ids * ((n_needed // len(filler_ids)) + 2))[:n_needed]
    idx = int(len(all_filler) * pos_frac)
    full = all_filler[:idx] + needle_ids + all_filler[idx:] + question_ids
    return full[:target_len], idx

def mrr_select(hidden, bs, d, k, bits, drift):
    seq_len, d_model = hidden.shape
    nb = seq_len // bs
    if nb <= k:
        return list(range(nb)), nb
    blocked = hidden[:nb*bs].reshape(nb, bs, d_model)
    sigs = blocked.mean(1)[:, :d]
    q = hidden[-1, :d]
    sm = sigs.abs().max() + 1e-10
    W = sigs / sm; q = q / (q.abs().max() + 1e-10)
    nl = 2**bits
    W = torch.round((W+1)/2*(nl-1))/(nl-1)*2-1 + torch.randn_like(W)*drift
    return torch.topk(W @ q, min(k, nb)).indices.tolist(), nb

def run_single(model, tokenizer, pos_frac, device):
    ids, needle_pos = build_context(tokenizer, CTX, pos_frac)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        chunk = 2048  # smaller chunks for 128K
        past = None; all_h = []
        for s in range(0, inp.shape[1], chunk):
            e = min(s+chunk, inp.shape[1])
            out = model(inp[:, s:e], past_key_values=past, output_hidden_states=True, use_cache=True)
            all_h.append(out.hidden_states[-1][0].cpu())  # move to CPU immediately
            past = out.past_key_values
            del out; torch.cuda.empty_cache()
        hidden = torch.cat(all_h, 0).to(device)
        del all_h, past; torch.cuda.empty_cache()

        sel, nb = mrr_select(hidden, B, D_SIG, K, QUANT_BITS, DRIFT_SIGMA)
        nblk = needle_pos // B
        found = nblk in sel
        del hidden; torch.cuda.empty_cache()

        gen = model.generate(inp, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(gen[0][inp.shape[1]:], skip_special_tokens=True).strip()
        del inp, gen; torch.cuda.empty_cache()
    return {"correct": EXPECTED in ans, "needle_found": found, "answer": ans[:50], "n_blocks": nb, "needle_block": nblk}

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(LOG, "w") as f: f.write("")
    log(f"NIAH 128K: {N_POSITIONS} positions")

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dk = "dtype" if tv >= (4,57) else "torch_dtype"

    log("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **{dk: torch.bfloat16}, device_map="cuda", trust_remote_code=True)
    model.eval()
    log(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    positions = np.linspace(0.05, 0.95, N_POSITIONS)
    results = []; t0 = time.time()
    for i, pos in enumerate(positions):
        try:
            r = run_single(model, tokenizer, pos, "cuda")
            results.append(r)
            log(f"  [{i+1}/{N_POSITIONS}] pos={pos:.2f} {'OK' if r['correct'] else 'MISS'} found={r['needle_found']}")
        except torch.cuda.OutOfMemoryError:
            log(f"  [{i+1}/{N_POSITIONS}] OOM"); torch.cuda.empty_cache(); gc.collect()
            results.append({"correct": False, "needle_found": False, "error": "OOM"})
        except Exception as e:
            log(f"  [{i+1}/{N_POSITIONS}] ERROR: {e}")
            results.append({"correct": False, "needle_found": False, "error": str(e)})
        if (i+1) % 5 == 0: gc.collect(); torch.cuda.empty_cache()

    n_valid = sum(1 for r in results if "error" not in r)
    acc = sum(1 for r in results if r.get("correct")) / max(n_valid, 1)
    recall = sum(1 for r in results if r.get("needle_found")) / max(n_valid, 1)
    log(f"\n128K: Accuracy={acc:.1%} Needle_recall={recall:.1%} Valid={n_valid}/{N_POSITIONS} Time={time.time()-t0:.0f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"context": CTX, "accuracy": acc, "needle_recall": recall, "n_valid": n_valid, "results": results}, f, indent=2)
    log(f"Saved: {RESULTS_FILE}")
    log("Done!")

if __name__ == "__main__":
    main()
