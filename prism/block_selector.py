"""
PRISM Block Selector — GPU-only mode.

Pure PyTorch block selection for KV cache. No photonic hardware needed.
Drop-in replacement for full KV cache attention that reduces memory
traffic by selecting only the top-k most relevant blocks.

This is the ALGORITHM that PRISM accelerates photonically.
On GPU, it already saves memory; with a photonic chip, it would be O(1).

Usage:
    from prism.block_selector import BlockSelector

    selector = BlockSelector(block_size=128, k=32, d_sig=32)

    # During prefill: build signatures from KV cache
    selector.build_signatures(kv_keys)  # [n_tokens, d_head]

    # During decode: select top-k blocks
    selected = selector.select(query)   # returns selected KV indices

    # Get selected keys/values for attention
    sel_keys, sel_values = selector.gather(kv_keys, kv_values, query)

    # Full pipeline: attention with block selection
    output = selector.block_sparse_attention(query, kv_keys, kv_values)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class BlockSelector:
    """
    GPU-only block selection for KV cache memory reduction.

    Partitions KV cache into blocks, computes mean-key signatures,
    and selects top-k blocks per query via inner product ranking.
    Compatible with any model — just pass key/value tensors.

    Args:
        block_size: Tokens per block. Default 128.
        k: Number of blocks to select. Default 32.
        d_sig: Signature projection dimension. Default 32.
        window: Recent tokens to always include. Default 256.
    """

    def __init__(self, block_size: int = 128, k: int = 32,
                 d_sig: int = 32, window: int = 256):
        self.block_size = block_size
        self.k = k
        self.d_sig = d_sig
        self.window = window
        self._signatures: Optional[torch.Tensor] = None
        self._projection: Optional[torch.Tensor] = None
        self._n_blocks: int = 0

    def build_signatures(self, keys: torch.Tensor) -> None:
        """
        Build block signatures from KV cache keys.

        Args:
            keys: [n_tokens, d_head] on any device
        """
        n_tokens, d_head = keys.shape
        self._n_blocks = n_tokens // self.block_size

        if self._n_blocks == 0:
            self._signatures = None
            return

        # Mean-key per block
        blocks = keys[:self._n_blocks * self.block_size].reshape(
            self._n_blocks, self.block_size, d_head)
        sigs = blocks.mean(dim=1)  # [n_blocks, d_head]

        # Random projection for dimensionality reduction
        if d_head > self.d_sig:
            if (self._projection is None or
                    self._projection.shape[0] != d_head or
                    self._projection.device != keys.device):
                self._projection = torch.randn(
                    d_head, self.d_sig, device=keys.device, dtype=keys.dtype)
                self._projection = F.normalize(self._projection, dim=0)
            self._signatures = sigs @ self._projection
        else:
            self._signatures = sigs
            self._projection = None

    @torch.no_grad()
    def select(self, query: torch.Tensor) -> torch.Tensor:
        """
        Select top-k block indices for given query.

        Args:
            query: [d_head] query vector

        Returns:
            [k] tensor of block indices
        """
        if self._signatures is None:
            return torch.arange(0, device=query.device)

        q = query.float()
        if self._projection is not None:
            q = q @ self._projection
        scores = self._signatures.float() @ q
        k = min(self.k, self._n_blocks)
        _, top_idx = torch.topk(scores, k)
        return top_idx

    @torch.no_grad()
    def gather(self, keys: torch.Tensor, values: torch.Tensor,
               query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select and gather top-k KV blocks + recent window.

        Args:
            keys: [n_tokens, d_head]
            values: [n_tokens, d_head]
            query: [d_head]

        Returns:
            (selected_keys, selected_values) — concatenated top-k blocks + window
        """
        n_tokens = keys.shape[0]
        top_idx = self.select(query)

        # Gather block tokens
        block_tokens = []
        for idx in top_idx:
            start = idx * self.block_size
            end = min(start + self.block_size, n_tokens)
            block_tokens.append(torch.arange(start, end, device=keys.device))

        # Recent window
        window_start = max(0, n_tokens - self.window)
        window_tokens = torch.arange(window_start, n_tokens, device=keys.device)

        # Combine and deduplicate
        all_tokens = torch.cat(block_tokens + [window_tokens])
        all_tokens = torch.unique(all_tokens, sorted=True)

        return keys[all_tokens], values[all_tokens]

    @torch.no_grad()
    def block_sparse_attention(self, query: torch.Tensor,
                                keys: torch.Tensor, values: torch.Tensor,
                                d_head: Optional[int] = None) -> torch.Tensor:
        """
        Full block-sparse attention: select blocks → compute exact attention.

        Args:
            query: [d_head] or [1, d_head]
            keys: [n_tokens, d_head]
            values: [n_tokens, d_head]

        Returns:
            [d_head] attention output
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        sel_keys, sel_values = self.gather(keys, values, query.squeeze(0))

        # Exact attention on selected tokens
        d = query.shape[-1]
        scores = (query @ sel_keys.T) / (d ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = weights @ sel_values

        return output.squeeze(0)

    @property
    def stats(self) -> dict:
        """Return selection statistics."""
        if self._signatures is None:
            return {}
        total_tokens = self._n_blocks * self.block_size
        selected_tokens = self.k * self.block_size + self.window
        return {
            'n_blocks': self._n_blocks,
            'total_tokens': total_tokens,
            'selected_tokens': min(selected_tokens, total_tokens),
            'traffic_reduction': f'{total_tokens / max(selected_tokens, 1):.1f}x',
            'memory_saved': f'{(1 - selected_tokens / total_tokens) * 100:.0f}%',
        }

    def __repr__(self):
        return (f"BlockSelector(block_size={self.block_size}, k={self.k}, "
                f"d_sig={self.d_sig}, window={self.window})")
