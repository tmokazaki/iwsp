"""
Live Subagent iWSP Experiment Record & Evaluator
===============================================

Records and reproduces the live Interpretable Weight-Space Programming (iWSP) experiment
conducted with an isolated subagent (`iwsp_scientist`, conversation
`9b0fa751-9d40-441c-b154-1ff45b22b60b`) with zero access to existing workspace code.

Tasks solved by the subagent from scratch in weight space:
- Iteration 1 (`abs_diff`): Compute |a - b| for [a, "-", b, "="] with a, b in 0..15 (256 cases).
- Iteration 2 (`multi_operator`): Dynamically switch between `a + b` (op="+") and |a - b| (op="-")
  for [a, op, b, "="] with a, b in 0..12 (338 cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from iwsp.core import APIScientistBackend, SandboxExecutor
from iwsp.transformers.architecture import TransparentTransformer

SUBAGENT_CONVERSATION_ID = "9b0fa751-9d40-441c-b154-1ff45b22b60b"

ITERATION_1_RESPONSE = r'''HYPOTHESIS: A 1-layer, 2-head, 2-neuron ReLU transformer with d_model=7 can compute exact |a - b| for [a, "-", b, "="] by encoding numeric values in W_E, distinguishing positions {0, 2, 3} via W_pos, routing `a` and `b` from positions 0 and 2 to position 3 ("=") via dedicated positional attention heads, computing ReLU(a - b) + ReLU(b - a) in a 2-neuron MLP, and projecting the scalar difference onto quadratic-envelope linear class logits (2*k*d - k^2) in W_U and b_U.
```python
def build_model():
    vocab = list(range(16)) + ["-", "="]
    d_model = 7
    dim_names = ["val", "pos0", "pos2", "pos3", "a_val", "b_val", "abs_diff"]
    VAL, POS0, POS2, POS3, A_VAL, B_VAL, ABS_DIFF = range(d_model)

    # Token embeddings: encode integer magnitude in `val`; operators have 0 magnitude.
    W_E = np.zeros((len(vocab), d_model))
    for k in range(16):
        W_E[k, VAL] = float(k)

    # Positional embeddings: mark positions 0 (operand a), 2 (operand b), and 3 ("=").
    W_pos = np.zeros((4, d_model))
    W_pos[0, POS0] = 1.0
    W_pos[2, POS2] = 1.0
    W_pos[3, POS3] = 1.0

    # Head 1: copy operand `a` from position 0 to position 3 ("=").
    W_Q_a = np.zeros((d_model, 1))
    W_Q_a[POS3, 0] = 1.0
    W_K_a = np.zeros((d_model, 1))
    W_K_a[POS0, 0] = 1.0
    W_V_a = np.zeros((d_model, 1))
    W_V_a[VAL, 0] = 1.0
    W_O_a = np.zeros((1, d_model))
    W_O_a[0, A_VAL] = 1.0
    head_copy_a = AttentionHeadWeights(
        name="copy_a",
        role="At pos3 ('='), attend sharply to pos0 and copy operand a into `a_val`.",
        W_Q=W_Q_a,
        W_K=W_K_a,
        W_V=W_V_a,
        W_O=W_O_a,
        causal=True,
        scale=50.0,
    )

    # Head 2: copy operand `b` from position 2 to position 3 ("=").
    W_Q_b = np.zeros((d_model, 1))
    W_Q_b[POS3, 0] = 1.0
    W_K_b = np.zeros((d_model, 1))
    W_K_b[POS2, 0] = 1.0
    W_V_b = np.zeros((d_model, 1))
    W_V_b[VAL, 0] = 1.0
    W_O_b = np.zeros((1, d_model))
    W_O_b[0, B_VAL] = 1.0
    head_copy_b = AttentionHeadWeights(
        name="copy_b",
        role="At pos3 ('='), attend sharply to pos2 and copy operand b into `b_val`.",
        W_Q=W_Q_b,
        W_K=W_K_b,
        W_V=W_V_b,
        W_O=W_O_b,
        causal=True,
        scale=50.0,
    )

    # 2-neuron ReLU MLP: |a - b| = ReLU(a - b) + ReLU(b - a).
    W_in = np.zeros((d_model, 2))
    W_in[A_VAL, 0] = 1.0
    W_in[B_VAL, 0] = -1.0
    W_in[A_VAL, 1] = -1.0
    W_in[B_VAL, 1] = 1.0
    b_in = np.zeros(2)

    W_out = np.zeros((2, d_model))
    W_out[0, ABS_DIFF] = 1.0
    W_out[1, ABS_DIFF] = 1.0

    mlp = MLPWeights(
        name="abs_diff_mlp",
        role="Compute |a - b| = ReLU(a_val - b_val) + ReLU(b_val - a_val) into `abs_diff`.",
        W_in=W_in,
        b_in=b_in,
        W_out=W_out,
        activation="relu",
        neuron_descriptions=[
            "Positive part: ReLU(a_val - b_val) = max(0, a - b)",
            "Negative part: ReLU(b_val - a_val) = max(0, b - a)",
        ],
    )

    layer0 = TransformerLayerWeights(
        name="layer0_route_and_abs_diff",
        role="Route a and b to '=' via positional attention, then compute |a - b| via ReLU MLP.",
        heads=[head_copy_a, head_copy_b],
        mlp=mlp,
    )

    # Unembedding: logits[k] = margin * (2 * k * d - k^2) = margin * (d^2 - (d - k)^2),
    # uniquely maximized at k = d = |a - b|.
    margin = 10.0
    W_U = np.zeros((d_model, 16))
    b_U = np.zeros(16)
    for k in range(16):
        W_U[ABS_DIFF, k] = 2.0 * margin * float(k)
        b_U[k] = -margin * float(k * k)

    return TransparentTransformer(
        name="abs_diff_transformer",
        description="Exact 1-layer iWSP transformer computing |a - b| for 4-token [a, '-', b, '='] inputs.",
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0],
        W_U=W_U,
        b_U=b_U,
        W_pos=W_pos,
        vocab=vocab,
        readout_mode="last_token",
    )
```'''

ITERATION_2_RESPONSE = r'''HYPOTHESIS: A 1-layer, 3-head, 3-neuron ReLU transformer with d_model=10 can dynamically switch between `a + b` and `|a - b|` by encoding token magnitude (`val`) and signed operator identity (`op_sign` = +1 for "+", -1 for "-") in W_E, routing `a`, `op`, and `b` from positions 0, 1, and 2 to position 3 ("=") via positional attention heads, gating one addition ReLU neuron and two absolute-difference ReLU neurons via big-M `op_flag` biases, and decoding classes 0..24 via quadratic-envelope logits in W_U and b_U.
```python
def build_model():
    vocab = list(range(13)) + ["+", "-", "="]
    d_model = 10
    dim_names = [
        "val",
        "op_sign",
        "pos0",
        "pos1",
        "pos2",
        "pos3",
        "a_val",
        "op_flag",
        "b_val",
        "result",
    ]
    VAL, OP_SIGN, POS0, POS1, POS2, POS3, A_VAL, OP_FLAG, B_VAL, RESULT = range(d_model)

    # Token embeddings: integer magnitude in `val`, signed operator (+1 for '+', -1 for '-') in `op_sign`.
    W_E = np.zeros((len(vocab), d_model))
    for k in range(13):
        W_E[k, VAL] = float(k)
    W_E[13, OP_SIGN] = 1.0   # "+"
    W_E[14, OP_SIGN] = -1.0  # "-"

    # Positional embeddings: mark positions 0 (a), 1 (op), 2 (b), and 3 ("=").
    W_pos = np.zeros((4, d_model))
    W_pos[0, POS0] = 1.0
    W_pos[1, POS1] = 1.0
    W_pos[2, POS2] = 1.0
    W_pos[3, POS3] = 1.0

    # Head 1: copy operand `a` from position 0 to position 3 ("=").
    W_Q_a = np.zeros((d_model, 1))
    W_Q_a[POS3, 0] = 1.0
    W_K_a = np.zeros((d_model, 1))
    W_K_a[POS0, 0] = 1.0
    W_V_a = np.zeros((d_model, 1))
    W_V_a[VAL, 0] = 1.0
    W_O_a = np.zeros((1, d_model))
    W_O_a[0, A_VAL] = 1.0
    head_copy_a = AttentionHeadWeights(
        name="copy_a",
        role="At pos3 ('='), attend sharply to pos0 and copy operand a into `a_val`.",
        W_Q=W_Q_a,
        W_K=W_K_a,
        W_V=W_V_a,
        W_O=W_O_a,
        causal=True,
        scale=50.0,
    )

    # Head 2: copy operator sign from position 1 to position 3 ("=").
    W_Q_op = np.zeros((d_model, 1))
    W_Q_op[POS3, 0] = 1.0
    W_K_op = np.zeros((d_model, 1))
    W_K_op[POS1, 0] = 1.0
    W_V_op = np.zeros((d_model, 1))
    W_V_op[OP_SIGN, 0] = 1.0
    W_O_op = np.zeros((1, d_model))
    W_O_op[0, OP_FLAG] = 1.0
    head_copy_op = AttentionHeadWeights(
        name="copy_op",
        role="At pos3 ('='), attend sharply to pos1 and copy operator sign (+1/-1) into `op_flag`.",
        W_Q=W_Q_op,
        W_K=W_K_op,
        W_V=W_V_op,
        W_O=W_O_op,
        causal=True,
        scale=50.0,
    )

    # Head 3: copy operand `b` from position 2 to position 3 ("=").
    W_Q_b = np.zeros((d_model, 1))
    W_Q_b[POS3, 0] = 1.0
    W_K_b = np.zeros((d_model, 1))
    W_K_b[POS2, 0] = 1.0
    W_V_b = np.zeros((d_model, 1))
    W_V_b[VAL, 0] = 1.0
    W_O_b = np.zeros((1, d_model))
    W_O_b[0, B_VAL] = 1.0
    head_copy_b = AttentionHeadWeights(
        name="copy_b",
        role="At pos3 ('='), attend sharply to pos2 and copy operand b into `b_val`.",
        W_Q=W_Q_b,
        W_K=W_K_b,
        W_V=W_V_b,
        W_O=W_O_b,
        causal=True,
        scale=50.0,
    )

    # Gated 3-neuron ReLU MLP:
    # - Neuron 0 (add):     ReLU(a + b + 15*op_flag - 15) -> a+b if op=='+' (+1), 0 if op=='-' (-1, since a+b<=24<30)
    # - Neuron 1 (sub_pos): ReLU(a - b - 10*op_flag - 10) -> max(0, a-b) if op=='-' (-1), 0 if op=='+' (+1, since a-b<=12<20)
    # - Neuron 2 (sub_neg): ReLU(b - a - 10*op_flag - 10) -> max(0, b-a) if op=='-' (-1), 0 if op=='+' (+1, since b-a<=12<20)
    W_in = np.zeros((d_model, 3))
    b_in = np.zeros(3)

    W_in[A_VAL, 0] = 1.0
    W_in[B_VAL, 0] = 1.0
    W_in[OP_FLAG, 0] = 15.0
    b_in[0] = -15.0

    W_in[A_VAL, 1] = 1.0
    W_in[B_VAL, 1] = -1.0
    W_in[OP_FLAG, 1] = -10.0
    b_in[1] = -10.0

    W_in[A_VAL, 2] = -1.0
    W_in[B_VAL, 2] = 1.0
    W_in[OP_FLAG, 2] = -10.0
    b_in[2] = -10.0

    W_out = np.zeros((3, d_model))
    W_out[0, RESULT] = 1.0
    W_out[1, RESULT] = 1.0
    W_out[2, RESULT] = 1.0

    mlp = MLPWeights(
        name="gated_multi_op_mlp",
        role="Switch between a + b and |a - b| using op_flag-gated ReLU branches.",
        W_in=W_in,
        b_in=b_in,
        W_out=W_out,
        activation="relu",
        neuron_descriptions=[
            "Gated addition: ReLU(a_val + b_val + 15*op_flag - 15) active only when op=='+'",
            "Gated positive diff: ReLU(a_val - b_val - 10*op_flag - 10) active only when op=='-'",
            "Gated negative diff: ReLU(b_val - a_val - 10*op_flag - 10) active only when op=='-'",
        ],
    )

    layer0 = TransformerLayerWeights(
        name="layer0_route_and_switch",
        role="Route operands and operator sign to '=', then compute gated a+b or |a-b|.",
        heads=[head_copy_a, head_copy_op, head_copy_b],
        mlp=mlp,
    )

    # Unembedding: logits[k] = margin * (2 * k * r - k^2) = margin * (r^2 - (r - k)^2) for k in 0..24.
    margin = 10.0
    W_U = np.zeros((d_model, 25))
    b_U = np.zeros(25)
    for k in range(25):
        W_U[RESULT, k] = 2.0 * margin * float(k)
        b_U[k] = -margin * float(k * k)

    return TransparentTransformer(
        name="multi_operator_transformer",
        description="Exact 1-layer iWSP transformer switching between a + b and |a - b| for [a, op, b, '='] inputs.",
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0],
        W_U=W_U,
        b_U=b_U,
        W_pos=W_pos,
        vocab=vocab,
        readout_mode="last_token",
    )
```'''


@dataclass
class SubagentEvaluationResult:
    iteration: int
    task_name: str
    hypothesis: str
    correct: int
    total: int
    accuracy: float
    stats: Dict[str, Any]
    errors: List[Tuple[Any, ...]]
    model: TransparentTransformer


def compile_subagent_model(raw_response: str) -> Tuple[str, TransparentTransformer]:
    """Parse and sandbox-compile a `build_model()` program generated by `iwsp_scientist`."""
    hypothesis, code = APIScientistBackend.parse_response(raw_response)
    sandbox = SandboxExecutor()
    build_fn, _ = sandbox.execute_and_extract(code, entrypoint="build_model")
    return hypothesis, build_fn()


def evaluate_abs_diff() -> SubagentEvaluationResult:
    """Evaluate Iteration 1 (`abs_diff`) across all 16 x 16 = 256 pairs."""
    hypothesis, model = compile_subagent_model(ITERATION_1_RESPONSE)
    correct = 0
    total = 0
    errors: List[Tuple[Any, ...]] = []
    for a in range(16):
        for b in range(16):
            total += 1
            pred = model.predict([a, "-", b, "="])
            target = abs(a - b)
            if pred == target:
                correct += 1
            elif len(errors) < 10:
                errors.append((a, "-", b, target, pred))
    return SubagentEvaluationResult(
        iteration=1,
        task_name="abs_diff",
        hypothesis=hypothesis,
        correct=correct,
        total=total,
        accuracy=correct / total,
        stats=model.parameter_stats(),
        errors=errors,
        model=model,
    )


def evaluate_multi_operator() -> SubagentEvaluationResult:
    """Evaluate Iteration 2 (`multi_operator`) across all 13 x 13 x 2 = 338 cases."""
    hypothesis, model = compile_subagent_model(ITERATION_2_RESPONSE)
    correct = 0
    total = 0
    errors: List[Tuple[Any, ...]] = []
    for op in ("+", "-"):
        for a in range(13):
            for b in range(13):
                total += 1
                pred = model.predict([a, op, b, "="])
                target = (a + b) if op == "+" else abs(a - b)
                if pred == target:
                    correct += 1
                elif len(errors) < 10:
                    errors.append((a, op, b, target, pred))
    return SubagentEvaluationResult(
        iteration=2,
        task_name="multi_operator",
        hypothesis=hypothesis,
        correct=correct,
        total=total,
        accuracy=correct / total,
        stats=model.parameter_stats(),
        errors=errors,
        model=model,
    )


def run_subagent_experiment() -> Tuple[SubagentEvaluationResult, SubagentEvaluationResult]:
    """Run and print the full live subagent iWSP evaluation summary."""
    res1 = evaluate_abs_diff()
    res2 = evaluate_multi_operator()
    for res in (res1, res2):
        print(f"=== Subagent Iteration {res.iteration}: {res.task_name} ===")
        print(f"Hypothesis: {res.hypothesis}")
        print(f"Accuracy: {res.correct}/{res.total} ({res.accuracy * 100:.1f}%)")
        print(
            f"Params: {res.stats['nonzero_params']}/{res.stats['total_params']} "
            f"(sparsity={res.stats['sparsity']*100:.1f}%), d_model={res.stats['d_model']}"
        )
        print(f"Errors: {res.errors}")
        print()
    print("=== Final Model Mechanistic Audit (Iteration 2) ===")
    print(res2.model.mechanistic_audit())
    return res1, res2


if __name__ == "__main__":
    run_subagent_experiment()
