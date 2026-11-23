"""
Command-Line Interface & Interactive Reproduction Demo for iWSP (arXiv:2611.18114)
=================================================================================

Usage:
    python3 -m iwsp.cli --mode all
    python3 -m iwsp.cli --mode algorithmic
    python3 -m iwsp.cli --mode nlp
    python3 -m iwsp.cli --mode brain
    python3 -m iwsp.cli --mode radio
    python3 -m iwsp.cli --mode loop --backend offline
    python3 -m iwsp.cli --mode loop --backend api --provider gemini
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from iwsp.brain.modalities import (
    BRAIN_MODALITIES,
    evaluate_brain_modality,
    get_brain_offline_trajectories,
)
from iwsp.core import APIScientistBackend, IWSPLoop, OfflineScientistBackend
from iwsp.radio import (
    build_beamforming_aoa_transformer,
    evaluate_beamforming_task,
    evaluate_radio_propagation,
    get_beamforming_offline_trajectory,
    get_radio_propagation_offline_trajectory,
    render_ascii_radio_heatmap,
)
from iwsp.transformers.algorithmic import (
    ALGORITHMIC_BENCHMARKS,
    evaluate_algorithmic_task,
    get_algorithmic_offline_trajectories,
)
from iwsp.transformers.nlp import (
    NLP_BENCHMARKS,
    evaluate_nlp_task,
    get_nlp_offline_trajectories,
)


def run_algorithmic_suite(verbose: bool = True) -> None:
    print("=" * 92)
    print("1. TRANSPARENT ALGORITHMIC TRANSFORMERS (Programmed in Weight-Space, 0 Gradient Steps)")
    print("=" * 92)
    print(f"{'Task':<12} | {'Accuracy':>9} | {'NonZero/Total':>15} | {'Sparsity':>8} | Mechanistic Summary")
    print("-" * 92)
    for task_name, builder in ALGORITHMIC_BENCHMARKS.items():
        acc, metrics, desc, model = evaluate_algorithmic_task(task_name, builder)
        nz_str = f"{metrics['nonzero_params']}/{metrics['total_params']}"
        sp_str = f"{metrics['sparsity']*100:.1f}%"
        short_desc = (desc[:43] + "...") if len(desc) > 46 else desc
        print(f"{task_name:<12} | {acc*100:>8.1f}% | {nz_str:>15} | {sp_str:>8} | {short_desc}")
    print()

    if verbose:
        add_model = ALGORITHMIC_BENCHMARKS["addition"]()
        print("--- Mechanistic Audit Sample: Addition Transformer ---")
        print(add_model.mechanistic_audit())
        trace = add_model.forward_with_trace([13, "+", 19, "="])
        pred = add_model.predict([13, "+", 19, "="])
        print(f"Sample Trace: [13, '+', 19, '='] -> Predicted={pred} (Final residual sum_val={float(trace['final_residual'][-1, 4]):.2f})")
        print()


def run_nlp_suite(verbose: bool = True) -> None:
    print("=" * 92)
    print("2. TRANSPARENT NLP CLASSIFICATION TRANSFORMERS (SST-2, TREC, MRPC, SNLI)")
    print("=" * 92)
    print(f"{'Benchmark':<12} | {'Accuracy':>9} | {'NonZero/Total':>15} | {'Sparsity':>8} | Mechanistic Summary")
    print("-" * 92)
    for task_name, builder in NLP_BENCHMARKS.items():
        acc, metrics, desc, model = evaluate_nlp_task(task_name, builder)
        nz_str = f"{metrics['nonzero_params']}/{metrics['total_params']}"
        sp_str = f"{metrics['sparsity']*100:.1f}%"
        short_desc = (desc[:43] + "...") if len(desc) > 46 else desc
        print(f"{task_name.upper():<12} | {acc*100:>8.1f}% | {nz_str:>15} | {sp_str:>8} | {short_desc}")
    print()


def run_brain_suite() -> None:
    print("=" * 92)
    print("3. TRANSPARENT BRAIN ENCODING MODELS (Visual 7T fMRI, Language fMRI, Auditory ECoG)")
    print("=" * 92)
    for modality in BRAIN_MODALITIES:
        mean_r, metrics, interp_summary, readout = evaluate_brain_modality(modality)
        print(f"[{modality.upper()}] Held-out Mean Pearson r = {mean_r:.4f} (Ridge alpha = {metrics['best_alpha']})")
        for reg_name, reg_r in metrics["region_r"].items():
            top_f = metrics["top_features"][reg_name]
            top_str = ", ".join(f"{fname} (w={w:+.3f})" for fname, w in top_f)
            print(f"   Region {reg_name:<22} | r = {reg_r:.4f} | Top Drivers: {top_str}")
        print()


def run_radio_suite(include_loop: bool = True) -> None:
    print("=" * 92)
    print("4. TRANSPARENT RADIO & WIRELESS SIMULATION MODELS (Beamforming AoA + 2D Propagation)")
    print("=" * 92)

    acc, bf_metrics, desc, bf_model = evaluate_beamforming_task()
    print(
        f"[ULA BEAMFORMING AoA] Accuracy = {acc*100:.1f}% "
        f"({bf_metrics['correct']}/{bf_metrics['total']}) | "
        f"NonZero/Total = {bf_metrics['nonzero_params']}/{bf_metrics['total_params']} "
        f"(Sparsity = {bf_metrics['sparsity']*100:.1f}%)"
    )
    print("--- Mechanistic Audit: ULA Beamforming AoA Transformer ---")
    print(bf_model.mechanistic_audit())
    print()

    mean_r, prop_metrics, _, readout = evaluate_radio_propagation()
    print(f"[2D RADIO PROPAGATION] Held-out Mean Pearson r = {mean_r:.4f} (Ridge alpha = {prop_metrics['best_alpha']})")
    for ch_name, ch_r in prop_metrics["region_r"].items():
        top_f = prop_metrics["top_features"][ch_name]
        top_str = ", ".join(f"{fname} (w={w:+.3f})" for fname, w in top_f)
        print(f"   Channel {ch_name:<24} | r = {ch_r:.4f} | Top Drivers: {top_str}")
    print()
    print(render_ascii_radio_heatmap(readout=readout, channel_idx=2, freq_ghz=2.4))
    print()

    if include_loop:
        trajectories = {}
        trajectories.update(get_beamforming_offline_trajectory())
        trajectories.update(get_radio_propagation_offline_trajectory())
        backend = OfflineScientistBackend(trajectories=trajectories)

        bf_loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
        bf_harv = bf_loop.run(
            task_name="beamforming_aoa",
            domain="RF Array Beamforming",
            task_description="Construct an 8-element ULA AoA beamforming TransparentTransformer.",
            evaluator_fn=evaluate_beamforming_task,
            entrypoint="build_model",
        )
        print(bf_harv.summary_table())
        print()

        prop_loop = IWSPLoop(backend=backend, max_iterations=3, target_score=0.995)
        prop_harv = prop_loop.run(
            task_name="radio_propagation",
            domain="RF 2D Propagation Encoding",
            task_description="Program interpretable RF propagation features extract_features(stimuli).",
            evaluator_fn=evaluate_radio_propagation,
            entrypoint="extract_features",
        )
        print(prop_harv.summary_table())
        print()


def run_iwsp_discovery_loops(backend_mode: str = "offline", provider: str = "auto") -> None:
    print("=" * 92)
    print(f"5. iWSP ITERATIVE SCIENTIFIC AGENT LOOP (Backend: {backend_mode.upper()})")
    print("=" * 92)

    if backend_mode == "offline":
        trajectories = {}
        trajectories.update(get_algorithmic_offline_trajectories())
        trajectories.update(get_nlp_offline_trajectories())
        trajectories.update(get_brain_offline_trajectories())
        trajectories.update(get_beamforming_offline_trajectory())
        trajectories.update(get_radio_propagation_offline_trajectory())
        backend = OfflineScientistBackend(trajectories=trajectories)
    else:
        backend = APIScientistBackend(provider=provider)

    # 1) Algorithmic iWSP Loop Demo (Addition & Parity)
    for algo_task in ("addition", "parity"):
        loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
        harvester = loop.run(
            task_name=algo_task,
            domain="Algorithmic Transformer",
            task_description=f"Construct a TransparentTransformer for {algo_task} directly in weight space.",
            evaluator_fn=lambda fn, tn=algo_task: evaluate_algorithmic_task(tn, fn),
            entrypoint="build_model",
        )
        print(harvester.summary_table())
        print()

    # 2) NLP iWSP Loop Demo (SST-2 Compositional Negation)
    nlp_loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
    nlp_harvester = nlp_loop.run(
        task_name="sst2",
        domain="NLP Classification Transformer",
        task_description="Construct a TransparentTransformer for SST-2 sentiment classification handling negation.",
        evaluator_fn=lambda fn: evaluate_nlp_task("sst2", fn),
        entrypoint="build_model",
    )
    print(nlp_harvester.summary_table())
    print()

    # 3) Brain Encoding iWSP Loop Demo (Visual Cortex & Language fMRI Confound Discovery)
    for brain_mod in ("visual_cortex", "language_fmri", "auditory_ecog"):
        steps = len(get_brain_offline_trajectories().get(brain_mod, [])) or 3
        b_loop = IWSPLoop(backend=backend, max_iterations=steps, target_score=0.99)
        b_harvester = b_loop.run(
            task_name=brain_mod,
            domain="Neuroscience Brain Encoding",
            task_description=f"Program interpretable feature space extract_features(stimuli) for {brain_mod}.",
            evaluator_fn=lambda fn, bm=brain_mod: evaluate_brain_modality(bm, fn),
            entrypoint="extract_features",
        )
        print(b_harvester.summary_table())
        print()

    # 4) Radio Simulation & Beamforming iWSP Loop Demo
    bf_loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
    bf_harvester = bf_loop.run(
        task_name="beamforming_aoa",
        domain="RF Array Beamforming",
        task_description="Construct an 8-element ULA AoA beamforming TransparentTransformer.",
        evaluator_fn=evaluate_beamforming_task,
        entrypoint="build_model",
    )
    print(bf_harvester.summary_table())
    print()

    prop_loop = IWSPLoop(backend=backend, max_iterations=3, target_score=0.995)
    prop_harvester = prop_loop.run(
        task_name="radio_propagation",
        domain="RF 2D Propagation Encoding",
        task_description="Program interpretable RF propagation features extract_features(stimuli).",
        evaluator_fn=evaluate_radio_propagation,
        entrypoint="extract_features",
    )
    print(prop_harvester.summary_table())
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce arXiv:2611.18114: Interpretable Weight-Space Programming (iWSP)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "algorithmic", "nlp", "brain", "radio", "loop"],
        default="all",
        help="Which reproduction suite to run (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["offline", "api"],
        default="offline",
        help="Scientist backend for the iWSP iterative loop (default: offline)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic", "gemini"],
        default="auto",
        help="LLM API provider when --backend=api",
    )
    args = parser.parse_args(argv)

    if args.mode in ("all", "algorithmic"):
        run_algorithmic_suite()
    if args.mode in ("all", "nlp"):
        run_nlp_suite()
    if args.mode in ("all", "brain"):
        run_brain_suite()
    if args.mode == "radio":
        run_radio_suite(include_loop=True)
    elif args.mode == "all":
        run_radio_suite(include_loop=False)
    if args.mode in ("all", "loop"):
        run_iwsp_discovery_loops(backend_mode=args.backend, provider=args.provider)

    return 0


if __name__ == "__main__":
    sys.exit(main())
