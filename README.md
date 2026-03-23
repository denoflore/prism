<h1 align="center">PRISM</h1>

<p align="center">
  <b>PRISM replaces O(N) KV cache scanning with O(1) photonic block selection.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-under%20review-lightgrey.svg" alt="arXiv: under review">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-ee4c2c.svg" alt="PyTorch">
</p>

<p align="center">
  <img src="assets/hero_v3.png?v=4" width="700"/>
</p>

---

## Why PRISM?

Long-context LLM inference is **memory-bound, not compute-bound**. Every decode step reads the entire KV cache from HBM — a wall that no amount of arithmetic scaling can break.

Existing block-selection methods (Quest, RocketKV, InfLLM) reduce the blocks *fetched*, but **still scan all N candidates** at O(N) cost. The scan itself becomes the bottleneck at long contexts.

**PRISM eliminates the scan entirely.** Read only top-k blocks → **O(1) selection + O(k) memory access**.

| | GPU–HBM Full Scan | GPU Block Selection | Dedicated ASIC | **PRISM** |
|---|:---:|:---:|:---:|:---:|
| Scan eliminated? | N/A (reads all) | No | Partially (parallelized) | **Yes** (broadcast) |
| Signature storage | None | GPU HBM | Local SRAM | MRR weight bank |
| Selection method | None (read all) | Read N sigs from HBM, sequential inner product | N-way parallel compute | Optical broadcast + MRR inner product |
| HBM read (selection) | 0 | N×d×2 bytes | 0 (local SRAM) | **0** (stored in MRR) |
| Selection latency | 0 | ~1-5 us (HBM-bound) | ~10-100 ns (SRAM) | **~9 ns** (optical O(1)) |
| Selection energy | 0 | ~4-16 uJ (HBM energy) | ~10-100 nJ (SRAM+MAC) | **~2.3 nJ** (laser+ADC) |
| Scaling with N | O(N) HBM read | O(N) HBM read | O(N) area, power, routing | **O(1)** passive split |
| Static power | 0 | 0 | ~W (SRAM leakage) | **~0** (Pockels EO) |
| Extra hardware | None | None | Dedicated chip | Photonic chip |

## Quick Start

```bash
git clone https://github.com/hyoseokp/PRISM.git
cd PRISM
pip install -e .

# Run demo — instant results, no GPU required
python demo.py
```

Output (1M context):
```
PRISM vs H100 Comparison Report (N=8192, d=32, k=32)
  Signature scan:   8.5 us (GPU) → ELIMINATED (PRISM)
  Selection:        8.5 us → 9 ns  (944x faster)
  Energy:           42 uJ → 2.3 nJ (18,000x less)
  HBM traffic:      512 KB scan + 2 MB fetch → 2 MB only (20% saved)
```

## Results

Tested on **Qwen2.5-7B**, block size B=128, signature dimension d=32.

| Metric | Value | Conditions |
|--------|-------|------------|
| Needle-block hit-rate | **100%** | 4K–64K tokens, k=32 |
| KV traffic reduction | **32x** | 128K context, k=32, B=128 |
| Retrieval heads | **>90%** | Qwen2.5-7B, tau=0.3 |
| Proxy recall@32 | **100%** | 8K context, mean-key, d=32 |
| Signed weight improvement | **+87%** | vs ReLU encoding |
| Dynamic selection energy | **2,290 pJ** | per query, optical core only |
| LongBench-v2 (3 domains) | **0% drop** | vs full attention, 4K |

> All GPU values estimated from H100 datasheet. PRISM values from device-physics simulation — no fabricated chip. Dynamic energy excludes TEC (~1W), amortized by query rate. See paper for full details.

## How It Works

```
Every B=128 tokens:   GPU computes block signature → programs MRR weights
Every decode step:    Query → light → broadcast → N inner products → top-k → fetch k blocks
                      \___ O(1) photonic ___/   \__ O(k) electronic __/
```

<p align="center">
  <img src="assets/fig_system_architecture_tikz.png" width="525"/>
</p>

**The physics does the math**: MRR transmission = multiplication, broadband photodetection = summation. No electronic MAC needed.

## GPU-Only Block Selection (no photonic chip needed)

PRISM's block selection algorithm also works as a **pure GPU memory optimizer** — no photonic hardware required. Similar to Quest/RocketKV but with mean-key signatures and random projection.

```python
from prism.block_selector import BlockSelector

selector = BlockSelector(block_size=128, k=32, d_sig=32, window=256)
selector.build_signatures(kv_keys)           # [n_tokens, d_head]
output = selector.block_sparse_attention(query, kv_keys, kv_values)

print(selector.stats)
# {'traffic_reduction': '30.1x', 'memory_saved': '97%', ...}
```

> **On GPU**: saves memory by reading only k blocks (same as Quest/RocketKV).
> **With PRISM photonic chip**: the selection itself becomes O(1) instead of O(N).

### How much does the scan cost? (measured on GPU)

```bash
python benchmarks/compare_selection_methods.py
```

| Context | N blocks | GPU scan (measured) | Scan-free* | Speedup |
|--------:|--------:|-------------------:|----------:|--------:|
| 128K | 1,024 | 1.1 us | 9 ns | 122x |
| 1M | 8,192 | 8.5 us | 9 ns | 944x |
| 10M | 81,920 | 80 us | 9 ns | 8,889x |

> \*Scan-free = theoretical lower bound if scan is fully eliminated (e.g., via photonic broadcast). No fabricated PRISM chip — 9 ns is a device-physics estimate. GPU scan values are real measurements.

## For LLM Engineers

**No photonic chip needed.** The simulator runs on any GPU/CPU:

```python
from prism.simulator import PRISMSimulator

# Works with ANY HuggingFace model (Qwen, Llama, Mistral, ...)
sim = PRISMSimulator(d_sig=32, k=32, block_size=128, bits=5, drift=0.01)
sim.register_signatures(kv_keys)      # [n_tokens, d_head]
top_indices = sim.select(query_vec)    # → k block indices
sim.report()                           # → stage-by-stage GPU vs PRISM comparison
```

### Profile the bottleneck on YOUR GPU

```bash
python benchmarks/profile_kv_scan.py --n_blocks 1024 --d 32
```

### Run NIAH benchmark

```bash
python benchmarks/run_niah.py --quick    # ~2 min, 16GB VRAM
python benchmarks/run_niah.py            # full 4K-64K evaluation
```

## For Photonic Computing Researchers

**KV cache block selection is a killer application for broadcast-and-weight photonic hardware.** Here's why your chip is a natural fit:

| Property of the workload | Why it matches B&W photonics |
|---|---|
| Query broadcast to all N candidates | = passive 1×N optical splitting |
| Block signatures are quasi-static (update every 128 tokens) | = MRR weight programming via EO DC bias |
| Only rank order matters (not exact values) | = 4-6 bit precision sufficient (relaxed DAC/ADC) |
| Latency-critical (per-token decode) | = O(1) optical transit vs O(N) electronic scan |
| Scales with context length | = advantage **grows** as N increases |

**If you have a B&W photonic chip**, this repo provides:

- **`prism/hw_sim/mrr_model.py`** — Lorentzian MRR model with Pockels EO tuning (X-cut TFLN, Q=10K, r33=30.9 pm/V). Plug in your own device parameters.
- **`prism/simulator.py`** — Full impairment pipeline (quantization, thermal drift, detector noise). Test your chip's specs against this workload.
- **`scripts/wdm_crosstalk.py`** — WDM channel isolation analysis for your channel spacing.
- **`benchmarks/compare_selection_methods.py`** — Measured GPU scan cost = the latency your chip needs to beat.

### Key device parameters (adapt to your platform)

| Parameter | PRISM (TFLN) | Your chip? |
|-----------|:---:|:---:|
| MRR Q | 10,000 | ? |
| Weight precision | 5 bit | ? |
| Tuning mechanism | Pockels EO (28.5 pm/V) | ? |
| Static power | ~0 (capacitive) | ? |
| Signed weights | Balanced PD (w ∈ [-1,+1]) | ? |
| WDM channels | d=32-128 | ? |

> **The question is not "can photonics do matrix multiply faster than GPU?" (probably not). The question is "where does photonic broadcast provide a structural advantage?" KV cache selection is that place.**

## Batch Serving: Where PRISM Shines

In batch serving, model weights are shared across users but **each user's KV signature scan competes for HBM bandwidth**. PRISM eliminates this competition entirely — signatures live in MRR weights, not HBM.

### Decode step breakdown (batch=128, 1M context, Qwen2.5-7B on H100)

| Stage | GPU Full Scan | Quest | PRISM (1 chip) |
|-------|:---:|:---:|:---:|
| HBM READ: model weights | 4,179 us | 4,179 us | 4,179 us |
| HBM READ: KV scan/read | **2,189,254 us** | — | — |
| HBM READ: signature scan | — | **8,567 us** | — |
| Photonic selection | — | — | **1,490 us** |
| HBM READ: KV fetch (k=32) | — | 8,567 us | 8,567 us |
| | | | |
| **Total HBM read** | **7,348 GB** | **71 GB** | **43 GB** |
| **Total time** | **2,193 ms** | **21.3 ms** | **14.3 ms** |
| **Batch throughput** | 58 tok/s | 6,009 tok/s | **8,951 tok/s** |
| **Per-user throughput** | 0.5 tok/s | 47 tok/s | **70 tok/s** |
| **Per-user latency** | 2,193 ms | 21.3 ms | **14.3 ms** |

> GPU Full Scan at batch=128 requires 7,348 GB HBM — not feasible on a single GPU. Shown for reference only.

### PRISM scales much slower than GPU

Quest's scan grows as O(N) — at long contexts, scan dominates the decode step.
PRISM (N_chip=1,024) pages through blocks when N > N_chip, growing as O(N/N_chip) but with a per-page cost of only 13 ns — orders of magnitude below HBM read cost.

| Batch=128 | Quest (ms) | PRISM (ms) | Improvement |
|-----------|:---:|:---:|:---:|
| 128K | 13.8 | 13.0 | 1.06x |
| 1M | 21.3 | 14.3 | **1.5x** |
| 10M | 98.3 | 27.7 | **3.5x** |
| 100M | 859 | 161.8 | **5.3x** |

> PRISM uses time-division multiplexing: one chip (N_chip=1,024) serves all 128 users by reprogramming MRR weights per user (~4 ns on TFLN Pockels EO) + photonic evaluation (~9 ns) = 13 ns/user/head. When context exceeds chip capacity, PRISM pages through ceil(N/N_chip) configurations. Adding parallel banks eliminates paging entirely. All GPU values from H100 datasheet; PRISM values from device-physics simulation (no fabricated chip).

## Energy Crossover

<p align="center">
  <img src="assets/fig_crossover_contour.png" width="400"/>
</p>

<p align="center"><b>Black line</b> = crossover. <b>Blue</b> = PRISM wins. <b>Red</b> = GPU wins. <b>★</b> = paper design points.<br>PRISM is energy-favorable at context ≥4K tokens.</p>

## Chip Design

<p align="center">
  <img src="assets/fig_chip_layout_tikz.png" width="450"/>
</p>

<p align="center">8x8 TFLN MRR weight bank. Scales to d=32, N=256 (single chip) or d=64, N=1024 (multi-chip).</p>

## Repository Structure

```
PRISM/
├── prism/                    # Core Python package (PyTorch/CUDA)
│   ├── simulator.py          #   Drop-in PRISM emulator for any LLM
│   ├── hw_sim/               #   MRR physics, noise, energy models
│   ├── kv_block/             #   Block partitioning & signatures
│   ├── similarity/           #   Broadcast-and-weight engine
│   ├── evaluation/           #   Recall, traffic metrics
│   └── retrieval_head/       #   Head identification
├── benchmarks/               # GPU profiler + NIAH benchmark
├── scripts/                  # Paper figure generation
├── results/                  # Pre-computed JSON results
├── paper/                    # LaTeX source & PDF
└── demo.py                   # One-command demo
```

## Citation

```bibtex
@article{park2026prism,
  title   = {PRISM: Breaking the O(n) Memory Wall in Long-Context
             LLM Inference via O(1) Photonic Block Selection},
  author  = {Park, Hyoseok and Park, Yeonsang},
  journal = {Under review},
  year    = {2026}
}
```

## Related Work

- **[Quest](https://arxiv.org/abs/2406.10774)** — Query-aware KV cache selection (Tang et al., 2024)
- **[RocketKV](https://github.com/NVlabs/RocketKV)** — Two-stage coarse-fine KV retrieval (NVlabs, 2025)
- **[InfLLM](https://arxiv.org/abs/2402.04617)** — CPU-offloaded KV cache with block retrieval (Xiao et al., 2024)
- **[MRR-AEF](https://arxiv.org/abs/2603.12934)** — MRR cascade for photonic softmax (Park & Park, 2026)
- **[KVTC](https://arxiv.org/abs/2511.01815)** — KV cache transform coding, 20x compression (NVIDIA, ICLR 2026)

## License

MIT — see [LICENSE](LICENSE).
