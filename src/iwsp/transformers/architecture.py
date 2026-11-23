"""
Transparent Transformer Engine for Interpretable Weight-Space Programming (iWSP)
================================================================================

Implements a glass-box Transformer architecture where every weight matrix, bias vector,
attention head, and residual stream dimension is explicitly named, annotated with its
mechanistic role, and constructed directly in code without gradient descent.

Supports:
- Named residual stream dimensions (`dim_names`) so every channel in `d_model` has a
  documented semantic meaning.
- Multi-head attention (`AttentionHeadWeights`) with explicit $W_Q, W_K, W_V, W_O$,
  optional causal masking, temperature scaling, and attention pattern recording.
- Two-layer MLPs (`MLPWeights`) with configurable activation (`relu`, `gelu`, `step`, `linear`)
  and explicit neuron-level annotations.
- Optional LayerNorm or direct unnormalized residual stream (standard in mechanistic
  weight-space circuits where exact numerical scales carry arithmetic values).
- Full mechanistic audit report (`model.mechanistic_audit()`) and activation trace
  inspection (`model.forward_with_trace(tokens)`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from iwsp.math_backend import gelu, np, relu, softmax


@dataclass
class AttentionHeadWeights:
    """
    Explicitly programmed weights for a single Attention Head.

    In residual-stream formulation:
    - $Q = X W_Q + b_Q$  where $W_Q \in \mathbb{R}^{d_{model} \times d_{head}}$
    - $K = X W_K + b_K$  where $W_K \in \mathbb{R}^{d_{model} \times d_{head}}$
    - $V = X W_V + b_V$  where $W_V \in \mathbb{R}^{d_{model} \times d_{head}}$
    - $\text{scores} = \frac{Q K^T}{\sqrt{d_{head}}} + M$
    - $\text{head\_out} = \text{softmax}(\text{scores}) V W_O + b_O$
      where $W_O \in \mathbb{R}^{d_{head} \times d_{model}}$
    """

    name: str
    role: str
    W_Q: Any  # (d_model, d_head)
    W_K: Any  # (d_model, d_head)
    W_V: Any  # (d_model, d_head)
    W_O: Any  # (d_head, d_model)
    b_Q: Optional[Any] = None  # (d_head,)
    b_K: Optional[Any] = None  # (d_head,)
    b_V: Optional[Any] = None  # (d_head,)
    b_O: Optional[Any] = None  # (d_model,)
    causal: bool = False
    scale: Optional[float] = None
    attention_bias_fn: Optional[Callable[[int], Any]] = None

    def __post_init__(self) -> None:
        self.W_Q = np.asarray(self.W_Q, dtype=np.float64)
        self.W_K = np.asarray(self.W_K, dtype=np.float64)
        self.W_V = np.asarray(self.W_V, dtype=np.float64)
        self.W_O = np.asarray(self.W_O, dtype=np.float64)
        if self.b_Q is not None:
            self.b_Q = np.asarray(self.b_Q, dtype=np.float64)
        if self.b_K is not None:
            self.b_K = np.asarray(self.b_K, dtype=np.float64)
        if self.b_V is not None:
            self.b_V = np.asarray(self.b_V, dtype=np.float64)
        if self.b_O is not None:
            self.b_O = np.asarray(self.b_O, dtype=np.float64)

    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Compute head output for input `x` of shape `(seq_len, d_model)`.
        Returns `(head_output, attn_weights)` where `head_output` is `(seq_len, d_model)`
        and `attn_weights` is `(seq_len, seq_len)`.
        """
        seq_len, _ = x.shape
        d_head = self.W_Q.shape[1]
        q = x @ self.W_Q
        if self.b_Q is not None:
            q = q + self.b_Q
        k = x @ self.W_K
        if self.b_K is not None:
            k = k + self.b_K
        v = x @ self.W_V
        if self.b_V is not None:
            v = v + self.b_V

        scale_factor = self.scale if self.scale is not None else (1.0 / math.sqrt(max(d_head, 1)))
        scores = (q @ k.T) * scale_factor

        if self.attention_bias_fn is not None:
            scores = scores + np.asarray(self.attention_bias_fn(seq_len), dtype=np.float64)

        if self.causal:
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        context = attn_weights @ v
        out = context @ self.W_O
        if self.b_O is not None:
            out = out + self.b_O
        return out, attn_weights

    def count_nonzero_params(self, tol: float = 1e-9) -> Tuple[int, int]:
        """Return `(nonzero_params, total_params)`."""
        mats = [self.W_Q, self.W_K, self.W_V, self.W_O]
        for b in (self.b_Q, self.b_K, self.b_V, self.b_O):
            if b is not None:
                mats.append(b)
        total = sum(m.size for m in mats)
        nz = sum(int((np.abs(m) > tol).sum()) for m in mats)
        return nz, total


@dataclass
class MLPWeights:
    """
    Explicitly programmed two-layer MLP block in a Transformer layer:
        $h = \sigma(X W_{in} + b_{in})$
        $\text{out} = h W_{out} + b_{out}$
    """

    name: str
    role: str
    W_in: Any  # (d_model, d_mlp)
    b_in: Any  # (d_mlp,)
    W_out: Any  # (d_mlp, d_model)
    b_out: Optional[Any] = None  # (d_model,)
    activation: str = "relu"
    neuron_descriptions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.W_in = np.asarray(self.W_in, dtype=np.float64)
        self.b_in = np.asarray(self.b_in, dtype=np.float64)
        self.W_out = np.asarray(self.W_out, dtype=np.float64)
        if self.b_out is not None:
            self.b_out = np.asarray(self.b_out, dtype=np.float64)
        else:
            self.b_out = np.zeros((self.W_out.shape[1],), dtype=np.float64)

    def forward(self, x: Any) -> Tuple[Any, Any]:
        """
        Compute MLP output for `x` of shape `(seq_len, d_model)`.
        Returns `(mlp_out, hidden_activations)`.
        """
        pre = (x @ self.W_in) + self.b_in
        if self.activation == "relu":
            h = relu(pre)
        elif self.activation == "gelu":
            h = gelu(pre)
        elif self.activation == "tanh":
            h = np.tanh(pre)
        elif self.activation == "step":
            h = (pre > 0.0) * 1.0
        elif self.activation == "linear":
            h = pre
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        out = (h @ self.W_out) + self.b_out
        return out, h

    def count_nonzero_params(self, tol: float = 1e-9) -> Tuple[int, int]:
        mats = [self.W_in, self.b_in, self.W_out, self.b_out]
        total = sum(m.size for m in mats)
        nz = sum(int((np.abs(m) > tol).sum()) for m in mats)
        return nz, total


@dataclass
class TransformerLayerWeights:
    """Single glass-box Transformer layer containing Attention Heads and an optional MLP."""

    name: str
    role: str
    heads: List[AttentionHeadWeights] = field(default_factory=list)
    mlp: Optional[MLPWeights] = None
    use_layernorm: bool = False


class TransparentTransformer:
    """
    A fully transparent, weight-space programmed Transformer model.

    Attributes:
        name: Human-readable model identifier.
        description: Full mechanistic overview of how the model operates.
        vocab: List of tokens (orNone if continuous/integer embeddings via custom table).
        d_model: Dimensionality of the residual stream.
        dim_names: Semantic names for each residual stream dimension `0..d_model-1`.
        W_E: Token embedding matrix of shape `(vocab_size, d_model)`.
        W_pos: Positional embedding matrix of shape `(max_seq_len, d_model)` (optional).
        layers: Sequence of `TransformerLayerWeights`.
        W_U: Unembedding / readout matrix of shape `(d_model, num_outputs)`.
        b_U: Unembedding bias of shape `(num_outputs,)` (optional).
    """

    def __init__(
        self,
        name: str,
        description: str,
        d_model: int,
        dim_names: List[str],
        W_E: Any,
        layers: List[TransformerLayerWeights],
        W_U: Any,
        b_U: Optional[Any] = None,
        W_pos: Optional[Any] = None,
        vocab: Optional[List[Any]] = None,
        readout_mode: str = "last_token",
    ) -> None:
        self.name = name
        self.description = description
        self.d_model = d_model
        self.dim_names = dim_names
        self.W_E = np.asarray(W_E, dtype=np.float64)
        self.layers = layers
        self.W_U = np.asarray(W_U, dtype=np.float64)
        self.b_U = (
            np.asarray(b_U, dtype=np.float64)
            if b_U is not None
            else np.zeros((self.W_U.shape[1],), dtype=np.float64)
        )
        self.W_pos = np.asarray(W_pos, dtype=np.float64) if W_pos is not None else None
        self.vocab = vocab or list(range(self.W_E.shape[0]))
        self.token_to_id: Dict[Any, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.readout_mode = readout_mode

    def encode_tokens(self, tokens: Sequence[Any]) -> List[int]:
        """Convert a sequence of token strings or integer IDs into vocabulary indices."""
        ids: List[int] = []
        for t in tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            elif isinstance(t, int) and 0 <= t < self.W_E.shape[0]:
                ids.append(t)
            else:
                # Unknown token falls back to index 0 (<pad>/<unk>) if available
                ids.append(0)
        return ids

    def forward_with_trace(self, tokens: Sequence[Any]) -> Dict[str, Any]:
        """
        Execute a forward pass on `tokens` and record all intermediate residual stream
        states, attention patterns, and MLP neuron activations for mechanistic inspection.
        """
        token_ids = self.encode_tokens(tokens)
        seq_len = len(token_ids)
        x = self.W_E[token_ids]  # (seq_len, d_model)
        if self.W_pos is not None:
            pos_slice = self.W_pos[:seq_len]
            x = x + pos_slice

        trace: Dict[str, Any] = {
            "tokens": list(tokens),
            "token_ids": token_ids,
            "embed_out": x.copy(),
            "layers": [],
        }

        for layer in self.layers:
            layer_info: Dict[str, Any] = {
                "name": layer.name,
                "role": layer.role,
                "heads": {},
                "mlp_hidden": None,
            }
            if layer.heads:
                attn_total = np.zeros((seq_len, self.d_model), dtype=np.float64)
                for head in layer.heads:
                    h_out, attn_w = head.forward(x)
                    attn_total = attn_total + h_out
                    layer_info["heads"][head.name] = {
                        "role": head.role,
                        "attn_weights": attn_w,
                        "head_out": h_out,
                    }
                x = x + attn_total

            if layer.mlp is not None:
                mlp_out, mlp_h = layer.mlp.forward(x)
                x = x + mlp_out
                layer_info["mlp_hidden"] = mlp_h
                layer_info["mlp_out"] = mlp_out

            layer_info["residual_out"] = x.copy()
            trace["layers"].append(layer_info)

        logits_seq = (x @ self.W_U) + self.b_U  # (seq_len, num_outputs)
        if self.readout_mode == "last_token":
            logits = logits_seq[-1]
        elif self.readout_mode == "mean_pool":
            logits = logits_seq.mean(axis=0)
        elif self.readout_mode == "all_tokens":
            logits = logits_seq
        else:
            logits = logits_seq[-1]

        trace["final_residual"] = x
        trace["logits_seq"] = logits_seq
        trace["logits"] = logits
        return trace

    def forward(self, tokens: Sequence[Any]) -> Any:
        """Compute model logits for `tokens`."""
        return self.forward_with_trace(tokens)["logits"]

    def predict(self, tokens: Sequence[Any]) -> Any:
        """Predict discrete class/token index (or sequence of indices for `all_tokens`)."""
        logits = self.forward(tokens)
        if self.readout_mode == "all_tokens":
            return logits.argmax(axis=-1)
        return int(logits.argmax(axis=-1))

    def parameter_stats(self, tol: float = 1e-9) -> Dict[str, Any]:
        """Compute total vs non-zero parameter counts across the entire glass-box Transformer."""
        total = self.W_E.size + self.W_U.size + self.b_U.size
        nz = (
            int((np.abs(self.W_E) > tol).sum())
            + int((np.abs(self.W_U) > tol).sum())
            + int((np.abs(self.b_U) > tol).sum())
        )
        if self.W_pos is not None:
            total += self.W_pos.size
            nz += int((np.abs(self.W_pos) > tol).sum())

        for layer in self.layers:
            for head in layer.heads:
                h_nz, h_tot = head.count_nonzero_params(tol)
                nz += h_nz
                total += h_tot
            if layer.mlp is not None:
                m_nz, m_tot = layer.mlp.count_nonzero_params(tol)
                nz += m_nz
                total += m_tot

        sparsity = 1.0 - (nz / max(total, 1))
        return {
            "nonzero_params": nz,
            "total_params": total,
            "sparsity": round(sparsity, 4),
            "d_model": self.d_model,
            "num_layers": len(self.layers),
        }

    def mechanistic_audit(self) -> str:
        """
        Generate a complete human-readable mechanistic audit of the programmed Transformer,
        explaining every residual dimension, attention head, and MLP neuron.
        """
        stats = self.parameter_stats()
        lines = [
            f"Model: {self.name}",
            f"Description: {self.description}",
            (
                f"Architecture: d_model={self.d_model}, layers={len(self.layers)}, "
                f"non-zero params={stats['nonzero_params']}/{stats['total_params']} "
                f"(sparsity={stats['sparsity']*100:.1f}%)"
            ),
            "Residual Stream Channels:",
        ]
        for idx, dname in enumerate(self.dim_names):
            lines.append(f"  dim[{idx}]: {dname}")
        for l_idx, layer in enumerate(self.layers):
            lines.append(f"Layer {l_idx} ({layer.name}): {layer.role}")
            for head in layer.heads:
                h_nz, h_tot = head.count_nonzero_params()
                lines.append(f"  [Head] {head.name} ({h_nz}/{h_tot} nz): {head.role}")
            if layer.mlp is not None:
                m_nz, m_tot = layer.mlp.count_nonzero_params()
                lines.append(
                    f"  [MLP]  {layer.mlp.name} ({m_nz}/{m_tot} nz, act={layer.mlp.activation}): "
                    f"{layer.mlp.role}"
                )
                for n_i, ndesc in enumerate(layer.mlp.neuron_descriptions):
                    lines.append(f"         neuron[{n_i}]: {ndesc}")
        return "\n".join(lines)
