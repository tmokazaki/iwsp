"""
Comprehensive Unit & Integration Tests for `iwsp` (arXiv:2611.18114 Reproduction)
=================================================================================
"""

from __future__ import annotations

import unittest

from iwsp.brain.modalities import (
    BRAIN_MODALITIES,
    evaluate_brain_modality,
    get_brain_offline_trajectories,
)
from iwsp.core import (
    APIScientistBackend,
    IWSPLoop,
    OfflineScientistBackend,
    SandboxExecutor,
)
from iwsp.transformers.algorithmic import (
    ALGORITHMIC_BENCHMARKS,
    build_addition_transformer,
    build_gcd_transformer,
    build_majority_transformer,
    build_max_transformer,
    build_parity_transformer,
    build_sorting_transformer,
    evaluate_algorithmic_task,
    get_algorithmic_offline_trajectories,
)
from iwsp.transformers.nlp import (
    NLP_BENCHMARKS,
    evaluate_nlp_task,
    get_nlp_offline_trajectories,
    tokenize_text,
)


class TestTransparentAlgorithmicTransformers(unittest.TestCase):
    """Verify 100% accuracy and glass-box mechanistic properties on all algorithmic tasks."""

    def test_addition_transformer_exact(self) -> None:
        model = build_addition_transformer(max_val=20)
        for a, b in [(0, 0), (1, 2), (7, 13), (15, 19), (20, 20)]:
            pred = model.predict([a, "+", b, "="])
            self.assertEqual(pred, a + b, f"Failed on {a} + {b}")
        trace = model.forward_with_trace([8, "+", 14, "="])
        self.assertAlmostEqual(float(trace["final_residual"][-1, 4]), 22.0, places=4)

    def test_sorting_transformer_exact(self) -> None:
        model = build_sorting_transformer(max_val=9, seq_len=5)
        seq = [7, 1, 9, 0, 4]
        pred = model.predict(seq)
        pred_list = [int(x) for x in (pred.tolist() if hasattr(pred, "tolist") else pred)]
        self.assertEqual(pred_list, [0, 1, 4, 7, 9])

    def test_parity_transformer_exact(self) -> None:
        model = build_parity_transformer(max_len=6)
        for bits in ([0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]):
            pred = model.predict(bits + ["="])
            self.assertEqual(pred, sum(bits) % 2)

    def test_gcd_transformer_exact(self) -> None:
        model = build_gcd_transformer(max_val=12)
        for a, b, expected in [(12, 8, 4), (9, 6, 3), (7, 11, 1), (10, 5, 5), (12, 12, 12)]:
            pred = model.predict([a, ",", b, "="])
            self.assertEqual(pred, expected)

    def test_maximum_and_majority_transformers(self) -> None:
        max_m = build_max_transformer(max_val=20)
        self.assertEqual(max_m.predict([4, 19, 2, 11, 7, "="]), 19)

        maj_m = build_majority_transformer(num_symbols=6)
        self.assertEqual(maj_m.predict([3, 1, 3, 5, 3, "="]), 3)

    def test_all_algorithmic_benchmarks_100_percent(self) -> None:
        for task_name, builder in ALGORITHMIC_BENCHMARKS.items():
            acc, metrics, _, model = evaluate_algorithmic_task(task_name, builder)
            self.assertEqual(acc, 1.0, f"Algorithmic task {task_name} did not achieve 100% accuracy")
            self.assertGreater(metrics["sparsity"], 0.5)
            audit_text = model.mechanistic_audit()
            self.assertIn("Residual Stream Channels:", audit_text)


class TestTransparentNLPTransformers(unittest.TestCase):
    """Verify compositional negation and question/inference routing in weight-space NLP Transformers."""

    def test_sst2_compositional_negation(self) -> None:
        model = NLP_BENCHMARKS["sst2"]()
        self.assertEqual(model.predict(tokenize_text("A brilliant and wonderful film .")), 1)
        self.assertEqual(model.predict(tokenize_text("A terrible and boring film .")), 0)
        # Single negation flips positive -> negative
        self.assertEqual(model.predict(tokenize_text("The film is not good and never funny .")), 0)
        # Double negation ("not bad", "never boring") flips negative -> positive
        self.assertEqual(model.predict(tokenize_text("The movie is not bad at all .")), 1)
        self.assertEqual(model.predict(tokenize_text("Never dull and never boring .")), 1)

    def test_all_nlp_benchmarks_100_percent(self) -> None:
        for task_name, builder in NLP_BENCHMARKS.items():
            acc, metrics, _, _ = evaluate_nlp_task(task_name, builder)
            self.assertEqual(acc, 1.0, f"NLP benchmark {task_name} failed to reach 100% accuracy")
            self.assertGreater(metrics["sparsity"], 0.7)


class TestTransparentBrainEncodingModels(unittest.TestCase):
    """Verify neuroscience brain encoding models and confound discovery across Vision, Language, and Audition."""

    def test_visual_cortex_dissociation(self) -> None:
        mean_r, metrics, _, _ = evaluate_brain_modality("visual_cortex")
        self.assertGreater(mean_r, 0.95)
        v1_top = [name for name, _ in metrics["top_features"]["V1_Early"]]
        v4_top = [name for name, _ in metrics["top_features"]["V4_IT_Ventral"]]
        # V1 must be driven by oriented edge energy; V4/IT by color warmth & center-surround
        self.assertIn("horiz_edge_energy", v1_top)
        self.assertIn("red_blue_warmth", v4_top)

    def test_language_fmri_confound_discovery(self) -> None:
        mean_r, metrics, _, _ = evaluate_brain_modality("language_fmri")
        self.assertGreater(mean_r, 0.95)
        scan_top_feat, scan_top_weight = metrics["top_features"]["Scan_Boundary_ROI"][0]
        self.assertEqual(scan_top_feat, "scan_edge_confound")
        self.assertGreater(scan_top_weight, 0.5)

    def test_auditory_ecog_feature_discovery(self) -> None:
        mean_r, metrics, _, _ = evaluate_brain_modality("auditory_ecog")
        self.assertGreater(mean_r, 0.95)
        onset_top_feat, _ = metrics["top_features"]["STG_Onset_Transient"][0]
        pitch_top_feat, _ = metrics["top_features"]["STG_Harmonic_Pitch"][0]
        self.assertEqual(onset_top_feat, "positive_spectral_flux_onset")
        self.assertEqual(pitch_top_feat, "harmonic_comb_modulation")


class TestIWSPAgentLoopAndSandbox(unittest.TestCase):
    """Verify the iWSP iterative loop, no-gradient enforcement, and response parsing."""

    def test_sandbox_forbids_gradient_descent(self) -> None:
        sandbox = SandboxExecutor()
        illegal_code = """
def build_model():
    loss.backward()
    return None
"""
        with self.assertRaises(ValueError):
            sandbox.execute_and_extract(illegal_code, entrypoint="build_model")

    def test_iwsp_iterative_improvement_trajectory(self) -> None:
        trajectories = {}
        trajectories.update(get_algorithmic_offline_trajectories())
        trajectories.update(get_brain_offline_trajectories())
        backend = OfflineScientistBackend(trajectories=trajectories)

        # Test addition trajectory (Iter 1 < 0.1 -> Iter 2 == 1.0)
        loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
        harvester = loop.run(
            task_name="addition",
            domain="Algorithmic",
            task_description="Addition in weight space",
            evaluator_fn=lambda fn: evaluate_algorithmic_task("addition", fn),
            entrypoint="build_model",
        )
        self.assertEqual(len(harvester.history), 2)
        self.assertLess(harvester.history[0].val_score, 0.5)
        self.assertEqual(harvester.history[1].val_score, 1.0)
        self.assertIsNotNone(harvester.best_trial)
        self.assertEqual(harvester.best_trial.val_score, 1.0)

        # Test language_fmri confound discovery trajectory (3 iterations monotonically improving)
        b_loop = IWSPLoop(backend=backend, max_iterations=3, target_score=0.99)
        b_harv = b_loop.run(
            task_name="language_fmri",
            domain="Brain Encoding",
            task_description="Language fMRI encoding",
            evaluator_fn=lambda fn: evaluate_brain_modality("language_fmri", fn),
            entrypoint="extract_features",
        )
        self.assertEqual(len(b_harv.history), 3)
        self.assertLess(b_harv.history[0].val_score, b_harv.history[1].val_score)
        self.assertLess(b_harv.history[1].val_score, b_harv.history[2].val_score)
        self.assertGreater(b_harv.best_trial.val_score, 0.95)

    def test_api_scientist_response_parser(self) -> None:
        sample = (
            "HYPOTHESIS: Use uniform attention over operands.\n"
            "```python\n"
            "def build_model():\n"
            "    return 42\n"
            "```"
        )
        hyp, code = APIScientistBackend.parse_response(sample)
        self.assertEqual(hyp, "Use uniform attention over operands.")
        self.assertIn("def build_model():", code)


class TestLiveSubagentExperiment(unittest.TestCase):
    """Verify models constructed live by the isolated `iwsp_scientist` subagent."""

    def test_subagent_abs_diff_and_multi_operator(self) -> None:
        from iwsp.subagent_experiment import evaluate_abs_diff, evaluate_multi_operator

        res1 = evaluate_abs_diff()
        self.assertEqual(res1.correct, 256)
        self.assertEqual(res1.total, 256)
        self.assertEqual(res1.accuracy, 1.0)
        self.assertGreater(res1.stats["sparsity"], 0.8)

        res2 = evaluate_multi_operator()
        self.assertEqual(res2.correct, 338)
        self.assertEqual(res2.total, 338)
        self.assertEqual(res2.accuracy, 1.0)
        self.assertGreater(res2.stats["sparsity"], 0.8)


class TestTransparentRadioModels(unittest.TestCase):
    """Verify phased-array beamforming Transformer and 2D radio propagation encoding models."""

    def test_beamforming_aoa_transformer_100_percent(self) -> None:
        from iwsp.radio import (
            AOA_ANGLES_DEG,
            build_beamforming_aoa_transformer,
            evaluate_beamforming_task,
            quantize_phase_tokens,
        )

        acc, metrics, _, model = evaluate_beamforming_task()
        self.assertEqual(acc, 1.0)
        self.assertEqual(metrics["correct"], metrics["total"])
        self.assertGreater(metrics["sparsity"], 0.85)
        # Spot-check +30 deg prediction directly
        idx_30 = AOA_ANGLES_DEG.index(30)
        tokens_30 = quantize_phase_tokens(30, phase_offset_deg=15.0) + ["="]
        self.assertEqual(model.predict(tokens_30), idx_30)
        self.assertIn("SpatialArrayGatheringLayer", model.mechanistic_audit())

    def test_radio_propagation_wall_crossing_and_confound_discovery(self) -> None:
        from iwsp.radio import (
            DEFAULT_WALLS,
            count_wall_crossings,
            evaluate_radio_propagation,
            render_ascii_radio_heatmap,
        )

        # Verify 2D ray-casting wall crossing geometry from Tx=(15, 20)
        n_conc, n_dry = count_wall_crossings((15.0, 20.0), (80.0, 80.0), DEFAULT_WALLS)
        self.assertGreaterEqual(n_conc, 1)
        self.assertGreaterEqual(n_dry, 1)

        mean_r, metrics, _, readout = evaluate_radio_propagation()
        self.assertGreater(mean_r, 0.99)
        for ch_name, r_val in metrics["region_r"].items():
            self.assertGreater(r_val, 0.95, f"Channel {ch_name} below threshold")
        ind_top = [name for name, _ in metrics["top_features"]["Industrial_Sector_RSSI"]]
        self.assertIn("industrial_anomaly_zone_confound", ind_top)
        self.assertIn("tx_sector_boresight_gain", ind_top)

        heatmap = render_ascii_radio_heatmap(readout=readout, channel_idx=2, freq_ghz=2.4)
        self.assertIn("T", heatmap)
        self.assertIn("#", heatmap)
        self.assertIn("X", heatmap)

    def test_radio_iwsp_iterative_loops(self) -> None:
        from iwsp.radio import (
            evaluate_beamforming_task,
            evaluate_radio_propagation,
            get_beamforming_offline_trajectory,
            get_radio_propagation_offline_trajectory,
        )

        trajectories = {}
        trajectories.update(get_beamforming_offline_trajectory())
        trajectories.update(get_radio_propagation_offline_trajectory())
        backend = OfflineScientistBackend(trajectories=trajectories)

        bf_loop = IWSPLoop(backend=backend, max_iterations=2, target_score=1.0)
        bf_harv = bf_loop.run(
            task_name="beamforming_aoa",
            domain="RF Array Beamforming",
            task_description="8-element ULA AoA",
            evaluator_fn=evaluate_beamforming_task,
            entrypoint="build_model",
        )
        self.assertEqual(len(bf_harv.history), 2)
        self.assertLess(bf_harv.history[0].val_score, 0.5)
        self.assertEqual(bf_harv.history[1].val_score, 1.0)

        prop_loop = IWSPLoop(backend=backend, max_iterations=3, target_score=0.995)
        prop_harv = prop_loop.run(
            task_name="radio_propagation",
            domain="RF Propagation",
            task_description="2D RF propagation encoding",
            evaluator_fn=evaluate_radio_propagation,
            entrypoint="extract_features",
        )
        self.assertEqual(len(prop_harv.history), 3)
        self.assertLess(prop_harv.history[0].val_score, prop_harv.history[1].val_score)
        self.assertLess(prop_harv.history[1].val_score, prop_harv.history[2].val_score)
        self.assertGreater(prop_harv.best_trial.val_score, 0.99)


if __name__ == "__main__":
    unittest.main()
