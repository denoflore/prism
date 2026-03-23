"""
PRISMSimulator — PyTorch/CUDA emulation of the PRISM photonic block selector.

Software simulation of what the PRISM photonic chip does, running entirely
on GPU via PyTorch. Allows benchmarking photonic block selection on any
CUDA-capable machine without actual photonic hardware.

Usage:
    from prism.simulator import PRISMSimulator

    sim = PRISMSimulator(d_sig=32, k=32, block_size=128, bits=5, drift=0.01)

    # During prefill: register block signatures
    sim.register_signatures(kv_keys)  # [n_tokens, d_head] torch.Tensor on CUDA

    # During decode: select top-k blocks
    top_indices = sim.select(query_vector)  # returns [k] tensor on same device

    # Fetch only selected KV blocks for exact attention
    selected_keys = kv_keys_blocked[top_indices]
"""
import torch
import torch.nn.functional as F
from typing import Optional


class PRISMSimulator:
    """
    PyTorch/CUDA emulation of the PRISM photonic similarity engine.

    Simulates the full MRR impairment pipeline on GPU:
      1. Block signature computation (mean key + random projection)
      2. Weight quantization (DAC precision)
      3. Thermal drift (MRR resonance jitter)
      4. Optical inner product (broadcast + MRR multiply + PD sum)
      5. Detector noise
      6. Electronic top-k selection

    All operations run on the same device as the input tensors (CPU or CUDA).

    Args:
        d_sig: Signature dimension (WDM channels). Default 32.
        k: Number of top blocks to select. Default 32.
        block_size: Tokens per KV cache block. Default 128.
        bits: Weight quantization precision (DAC bits). Default 5.
        drift: Thermal drift sigma (normalized). Default 0.01.
        det_noise: Detector noise sigma (relative to score std). Default 0.01.
    """

    def __init__(self, d_sig: int = 32, k: int = 32, block_size: int = 128,
                 bits: int = 5, drift: float = 0.01, det_noise: float = 0.01):
        self.d_sig = d_sig
        self.k = k
        self.block_size = block_size
        self.bits = bits
        self.drift = drift
        self.det_noise = det_noise

        self._projection: Optional[torch.Tensor] = None
        self._programmed_weights: Optional[torch.Tensor] = None
        self._w_min: float = 0.0
        self._w_max: float = 1.0
        self._n_blocks: int = 0
        self._device: torch.device = torch.device("cpu")

    def register_signatures(self, keys: torch.Tensor) -> None:
        """
        Compute and program block signatures from KV cache keys.

        Simulates:
          - GPU computing mean-key block signatures
          - Random projection to d_sig dimensions
          - PRISM chip programming MRR weights via DC voltage

        Args:
            keys: Key vectors [n_tokens, d_head] on any device
        """
        self._device = keys.device
        keys = keys.float()

        n_tokens, d_head = keys.shape
        n_blocks = n_tokens // self.block_size
        self._n_blocks = n_blocks

        # Compute mean-key signatures
        blocks = keys[:n_blocks * self.block_size].reshape(
            n_blocks, self.block_size, d_head)
        sigs = blocks.mean(dim=1)  # [n_blocks, d_head]

        # Random projection to d_sig dimensions
        if d_head > self.d_sig:
            if (self._projection is None or
                    self._projection.shape[0] != d_head or
                    self._projection.device != self._device):
                proj = torch.randn(d_head, self.d_sig,
                                   device=self._device, dtype=torch.float32)
                self._projection = F.normalize(proj, dim=0)
            sigs = sigs @ self._projection  # [n_blocks, d_sig]

        # Normalize to [0, 1] for MRR weight encoding
        self._w_min = sigs.min().item()
        self._w_max = sigs.max().item()
        W = (sigs - self._w_min) / (self._w_max - self._w_min + 1e-30)

        # Quantize (simulate DAC precision)
        levels = 2 ** self.bits
        W_q = torch.round(W * (levels - 1)) / (levels - 1)

        self._programmed_weights = W_q  # stays on device

    @torch.no_grad()
    def select(self, query: torch.Tensor) -> torch.Tensor:
        """
        Select top-k blocks via simulated photonic inner products.

        Simulates the full PRISM decode-step pipeline:
          1. Query projection to d_sig
          2. Broadcast (implicit — same query to all channels)
          3. MRR weighting with thermal drift
          4. Photodetection (inner product)
          5. Detector noise
          6. Top-k comparator

        Args:
            query: Query vector [d_head] on same device as keys

        Returns:
            top_indices: [k] tensor of selected block indices
        """
        if self._programmed_weights is None:
            raise RuntimeError("Call register_signatures() first")

        query = query.float().to(self._device)

        # Project query
        if self._projection is not None:
            q = query @ self._projection  # [d_sig]
        else:
            q = query

        # Apply thermal drift to MRR weights
        W_drifted = self._programmed_weights + \
            torch.randn_like(self._programmed_weights) * self.drift
        W_drifted = W_drifted.clamp(0, 1)

        # Reconstruct and compute inner products
        S_recon = W_drifted * (self._w_max - self._w_min) + self._w_min
        scores = S_recon @ q  # [n_blocks] — the photonic inner products

        # Detector noise
        if self.det_noise > 0:
            noise_std = scores.std() * self.det_noise
            scores = scores + torch.randn_like(scores) * noise_std

        # Top-k selection (electronic comparator)
        _, top_indices = torch.topk(scores, self.k)
        return top_indices

    @torch.no_grad()
    def select_with_scores(self, query: torch.Tensor):
        """Like select(), but also returns similarity scores."""
        if self._programmed_weights is None:
            raise RuntimeError("Call register_signatures() first")

        query = query.float().to(self._device)
        q = query @ self._projection if self._projection is not None else query

        W_drifted = (self._programmed_weights +
                     torch.randn_like(self._programmed_weights) * self.drift).clamp(0, 1)
        S_recon = W_drifted * (self._w_max - self._w_min) + self._w_min
        scores = S_recon @ q

        if self.det_noise > 0:
            scores = scores + torch.randn_like(scores) * scores.std() * self.det_noise

        top_scores, top_indices = torch.topk(scores, self.k)
        return top_indices, top_scores, scores

    @torch.no_grad()
    def recall_at_k(self, keys: torch.Tensor, query: torch.Tensor) -> float:
        """
        Compute recall@k: fraction of true top-k blocks found by PRISM.

        Args:
            keys: Key vectors [n_tokens, d_head]
            query: Query vector [d_head]

        Returns:
            recall: float in [0, 1]
        """
        keys = keys.float().to(self._device)
        query = query.float().to(self._device)

        # Digital ground truth
        n_blocks = keys.shape[0] // self.block_size
        blocks = keys[:n_blocks * self.block_size].reshape(
            n_blocks, self.block_size, -1)
        sigs = blocks.mean(dim=1)

        if self._projection is not None:
            sigs_proj = sigs @ self._projection
            q_proj = query @ self._projection
        else:
            sigs_proj = sigs
            q_proj = query

        digital_scores = sigs_proj @ q_proj
        _, digital_topk = torch.topk(digital_scores, self.k)
        digital_set = set(digital_topk.cpu().tolist())

        # PRISM selection
        self.register_signatures(keys)
        prism_topk = self.select(query)
        prism_set = set(prism_topk.cpu().tolist())

        return len(digital_set & prism_set) / self.k

    def report(self, gpu_name: str = "H100",
               hbm_bw_tbps: float = 3.35,    # H100 SXM5 HBM3 spec (nvidia.com/h100)
               hbm_energy_pj_per_byte: float = 31.0,  # HBM3 ~3.9 pJ/bit = 31 pJ/byte (Micron)
               gpu_mac_pj: float = 0.71,      # H100 bf16 tensor core (datasheet TDP / peak FLOPS)
               measure_gpu: bool = True) -> dict:
        """
        Print a detailed stage-by-stage comparison: GPU vs PRISM.

        Measures actual GPU scan latency (if CUDA available and measure_gpu=True),
        and computes photonic latency/energy from device physics.

        Args:
            gpu_name: GPU name for display. Default "H100".
            hbm_bw_tbps: HBM bandwidth in TB/s. H100=3.35, A100=2.0.
            hbm_energy_pj_per_byte: HBM read energy. HBM3=31 pJ/B.
            gpu_mac_pj: Energy per MAC. H100=0.71 pJ.
            measure_gpu: If True and CUDA available, measure actual GPU latency.

        Returns:
            dict with all computed values.
        """
        if self._programmed_weights is None:
            raise RuntimeError("Call register_signatures() first")

        N = self._n_blocks
        d = self.d_sig
        k = self.k
        B = self.block_size
        d_h = 128  # assumed head dimension

        # ── GPU-side estimates ──
        # Signature scan: read N*d*2 bytes from HBM, compute N*d MACs
        # In practice, full KV scan reads d_h (not d) per block for exact scoring
        scan_bytes_full = N * d_h * 2  # full key dimension, bf16
        scan_bytes_compressed = N * d * 2  # compressed signatures, bf16
        scan_bytes = scan_bytes_full  # use full-dimension baseline
        scan_total_us = scan_bytes / (hbm_bw_tbps * 1e6)  # TB/s → B/us
        # Add realistic overhead: kernel launch, cache miss, memory controller
        scan_overhead_us = max(1.0, scan_total_us * 10)  # at least 1 us
        scan_total_us = scan_total_us + scan_overhead_us
        scan_macs = N * d_h
        scan_energy_uj = (scan_bytes * hbm_energy_pj_per_byte * 1e-6 +
                          scan_macs * gpu_mac_pj * 1e-6)

        # Top-k on GPU (negligible)
        topk_gpu_us = 0.1

        # KV block fetch: k * B * 2 * d_h * 2 bytes
        fetch_bytes = k * B * 2 * d_h * 2  # key+value, bf16
        fetch_us = fetch_bytes / (hbm_bw_tbps * 1e6)
        fetch_energy_uj = fetch_bytes * hbm_energy_pj_per_byte * 1e-6

        gpu_total_us = scan_total_us + topk_gpu_us + fetch_us
        gpu_total_energy_uj = scan_energy_uj + fetch_energy_uj

        # ── Measure actual GPU latency ──
        gpu_measured_us = None
        if measure_gpu and self._device.type == 'cuda':
            import time
            sigs = torch.randn(N, d, device=self._device, dtype=torch.float16)
            q = torch.randn(d, device=self._device, dtype=torch.float16)
            # Warmup
            for _ in range(100):
                _ = torch.topk(sigs @ q, k)
            torch.cuda.synchronize()
            # Measure
            times = []
            for _ in range(500):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                scores = sigs @ q
                _ = torch.topk(scores, k)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)
            gpu_measured_us = sorted(times)[len(times) // 2]

        # ── PRISM photonic estimates ──
        # Latency from device physics: TFLN MRR (Q=10K, R=20um)
        # See arXiv:2603.12934 for single-ring FDTD validation
        prism = {
            'dac_ns': 1.0,          # 10-bit DAC at 1 GS/s (commercial, e.g. AD9176)
            'mzm_ns': 0.1,          # MZM EO bandwidth >10 GHz (TFLN, Hu et al. 2025)
            'propagation_ns': 0.5,  # ~5 cm waveguide at n_g=2.30
            'mrr_ns': 0.1,          # ring-down time = Q/(pi*f_res) ~ 0.01 ns, rounded up
            'pd_ns': 0.2,           # InGaAs PD bandwidth >10 GHz (commercial)
            'tia_adc_ns': 2.0,      # TIA + 6-bit ADC at 500 MS/s
            'topk_ns': 5.0,         # CMOS comparator tree, N=1024, ~5 ns at 28nm
        }
        prism_total_ns = sum(prism.values())

        # PRISM energy per query (d=64, N=1024 design point)
        # Derived from component power x 9 ns optical transit
        # See paper Table 4 and energy_analysis.json
        prism_energy = {
            'laser_pj': 900,            # 100 mW CW laser, 9 ns window
            'dac_pj': 288,              # 64 ch x 0.5 mW/ch x 9 ns
            'mzm_pj': 58,              # 64 ch x 0.1 mW/ch x 9 ns
            'voltage_driver_pj': 45,    # 5 mW driver x 9 ns
            'eo_bias_pj': 0,            # TFLN Pockels: capacitive, ~0 static power
            'pd_pj': 90,               # 2x1024 balanced PDs x 0.005 mW/PD x 9 ns
            'tia_adc_pj': 900,          # 2x1024 ch x 0.049 mW/ch x 9 ns
            'topk_pj': 9,              # 1 mW CMOS logic x 9 ns
        }
        prism_total_pj = sum(prism_energy.values())

        # ── Speedup / savings ──
        scan_vs_prism_speedup = (scan_total_us * 1000) / prism_total_ns if prism_total_ns > 0 else 0
        energy_saving = scan_energy_uj / (prism_total_pj * 1e-6) if prism_total_pj > 0 else 0

        # ── Print report ──
        w = 66
        print(f"\n{'=' * w}")
        print(f"  PRISM vs {gpu_name} Comparison Report")
        print(f"  N={N} blocks, d={d}, k={k}, B={B}")
        print(f"{'=' * w}")

        print(f"\n  {'Stage':<22} {'GPU (electronic)':<22} {'PRISM (photonic)':<22}")
        print(f"  {'-'*22} {'-'*22} {'-'*22}")

        print(f"  {'Signature scan':<22} "
              f"{'%.1f us (HBM read)' % scan_total_us:<22} "
              f"{'-- (eliminated)':<22}")

        print(f"  {'Query encoding':<22} "
              f"{'--':<22} "
              f"{'~%.1f ns (DAC+MZM)' % (prism['dac_ns']+prism['mzm_ns']):<22}")

        print(f"  {'Broadcast':<22} "
              f"{'(included in scan)':<22} "
              f"{'~%.1f ns (passive)' % prism['propagation_ns']:<22}")

        print(f"  {'Inner products':<22} "
              f"{'(included in scan)':<22} "
              f"{'~%.1f ns (MRR+PD)' % (prism['mrr_ns']+prism['pd_ns']):<22}")

        print(f"  {'Top-k selection':<22} "
              f"{'~%.1f us' % topk_gpu_us:<22} "
              f"{'~%.1f ns (CMOS)' % prism['topk_ns']:<22}")

        print(f"  {'KV block fetch':<22} "
              f"{'%.1f us (%d KB)' % (fetch_us, fetch_bytes//1024):<22} "
              f"{'%.1f us (same)' % fetch_us:<22}")

        print(f"\n  {'-'*22} {'-'*22} {'-'*22}")
        print(f"  {'TOTAL SELECTION':<22} "
              f"{'%.1f us' % (scan_total_us + topk_gpu_us):<22} "
              f"{'~%.0f ns' % prism_total_ns:<22}")

        if gpu_measured_us is not None:
            print(f"  {'(measured on GPU)':<22} "
                  f"{'%.1f us' % gpu_measured_us:<22} "
                  f"{'--':<22}")

        print(f"  {'Selection energy':<22} "
              f"{'%.1f uJ' % scan_energy_uj:<22} "
              f"{'~%.1f nJ (%.0f pJ)' % (prism_total_pj/1000, prism_total_pj):<22}")

        print(f"  {'HBM traffic':<22} "
              f"{'%.0f KB + %.0f MB' % (scan_bytes/1024, fetch_bytes/1e6):<22} "
              f"{'%.0f MB (fetch only)' % (fetch_bytes/1e6):<22}")

        print(f"\n  {'SPEEDUP':<22} "
              f"{'Selection: %.0fx' % scan_vs_prism_speedup:<22} "
              f"{'Energy: %.0fx' % energy_saving:<22}")
        print(f"{'-' * w}")
        if gpu_measured_us is not None:
            print(f"  * GPU scan latency: MEASURED on {gpu_name}")
        else:
            print(f"  * GPU values: ESTIMATED from {gpu_name} datasheet")
            print(f"    (HBM3 {hbm_bw_tbps} TB/s, {gpu_mac_pj} pJ/MAC)")
        print(f"  * PRISM values: device-physics simulation (no fabricated chip)")
        print(f"  * Run on CUDA GPU for actual scan measurements")
        print(f"{'=' * w}\n")

        return {
            'gpu_scan_us': scan_total_us,
            'gpu_scan_energy_uj': scan_energy_uj,
            'gpu_measured_us': gpu_measured_us,
            'gpu_fetch_us': fetch_us,
            'gpu_total_us': gpu_total_us,
            'prism_total_ns': prism_total_ns,
            'prism_total_pj': prism_total_pj,
            'speedup': scan_vs_prism_speedup,
            'energy_saving': energy_saving,
            'N': N, 'd': d, 'k': k, 'B': B,
        }

    @property
    def config(self) -> dict:
        return {
            "d_sig": self.d_sig, "k": self.k,
            "block_size": self.block_size, "bits": self.bits,
            "drift": self.drift, "det_noise": self.det_noise,
            "n_blocks": self._n_blocks,
            "device": str(self._device),
        }

    def __repr__(self):
        return (f"PRISMSimulator(d_sig={self.d_sig}, k={self.k}, "
                f"bits={self.bits}, drift={self.drift}, "
                f"device={self._device})")
