# `iwsp`: Interpretable Weight-Space Programming (arXiv:2611.18114v1)

A comprehensive Python package reproducing **"Interpretable weight-space programming builds transparent transformers and brain encoding models"** (*Yash Mehta, Chandan Singh, Weijia Xu, Richard J. Antonello, Adeen Flinker, Nima Mesgarani, Michael F. Bonner, Nebojsa Jojic, Jianfeng Gao*, [arXiv:2611.18114v1](https://arxiv.org/html/2611.18114v1)).

---

## Core Paradigm: What is iWSP?

Instead of optimizing neural network parameters via gradient descent (backpropagation), **Interpretable Weight-Space Programming (iWSP)** treats a model's weights and feature representations as an explicit program written by an LLM scientist agent:
1. **Hypothesize**: The agent reasons about the task and proposes a mechanistic design where every residual-stream dimension, attention head ($W_Q, W_K, W_V, W_O$), MLP neuron ($W_{in}, b_{in}, W_{out}, b_{out}$), or cortical feature has a human-readable role.
2. **Program Weights in Code**: The agent writes explicit Python code constructing the weights directly (with zero `.backward()` or gradient optimizer steps).
3. **Evaluate & Refine**: The code is executed in a sandboxed harness against a validation metric (accuracy for Transformers; cross-validated Ridge Regression Pearson $r$ for Brain Encoding), and the agent iteratively refines its weight program across trials.

---

## Package Structure

```text
iwsp/
├── pyproject.toml
├── README.md
├── docs/
│   └── ARCHITECTURE.md         # Detailed math, residual stream maps, circuit diagrams & parameter tables
├── src/iwsp/
│   ├── __init__.py
│   ├── core.py                 # IWSPLoop, SandboxExecutor (no-gradient enforcement), Offline & Live API Backends
│   ├── math_backend.py         # Unified NumPy / Pure-Python Linear Algebra & Tensor Engine (zero-dependency fallback)
│   ├── cli.py                  # Interactive CLI reproduction runner (`python3 -m iwsp.cli`)
│   ├── subagent_experiment.py  # Live isolated subagent iWSP experiment record (abs_diff & multi_operator)
│   ├── transformers/
│   │   ├── __init__.py
│   │   ├── architecture.py     # Glass-box TransparentTransformer, AttentionHeadWeights, MLPWeights, mechanistic audit
│   │   ├── algorithmic.py      # Weight-programmed Transformers for Addition, Sorting, Parity, GCD, Max, Majority
│   │   └── nlp.py              # Weight-programmed Transformers for SST-2, TREC, MRPC, and SNLI
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── ridge.py            # Cross-validated RidgeEncodingReadout & region-wise feature attribution
│   │   └── modalities.py       # Visual Cortex (7T fMRI/NSD), Language fMRI (confound discovery), Auditory ECoG
│   └── radio/
│       ├── __init__.py
│       ├── beamforming.py      # 8-element ULA Phased-Array AoA / Beamforming Transformer
│       └── propagation.py      # 2D RF propagation encoding (wall ray-casting, 2-ray reflection, anomaly discovery, ASCII heatmap)
└── tests/
    └── test_iwsp.py            # Full unit and integration test suite
```

---

## Quick Start

### 1. Run Full Reproduction Demo (Offline / No API Key Required)

```bash
PYTHONPATH=src python3 -m iwsp.cli --mode all
```

You can also run individual suites:

```bash
# Algorithmic Transformers (Addition, Sorting, Parity, GCD, Max, Majority)
PYTHONPATH=src python3 -m iwsp.cli --mode algorithmic

# NLP Classification Transformers (SST-2, TREC, MRPC, SNLI)
PYTHONPATH=src python3 -m iwsp.cli --mode nlp

# Neuroscience Brain Encoding Models (Visual Cortex, Language fMRI, Auditory ECoG)
PYTHONPATH=src python3 -m iwsp.cli --mode brain

# Radio Simulation & Array Beamforming Models (AoA Transformer + 2D Propagation + Heatmap)
PYTHONPATH=src python3 -m iwsp.cli --mode radio

# iWSP Iterative Scientific Discovery Loop
PYTHONPATH=src python3 -m iwsp.cli --mode loop --backend offline
```

### 2. Run with a Live LLM API (OpenAI / Anthropic / Gemini)

Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` and pass `--backend api`:

```bash
GEMINI_API_KEY="your-key" PYTHONPATH=src python3 -m iwsp.cli --mode loop --backend api --provider gemini
```

### 3. Run Unit Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## Architecture Documentation

Detailed mathematical formulations, residual stream channel allocations, Attention/MLP circuit diagrams (Mermaid), and parameter/sparsity summary tables for all 17 models across Algorithmic, Subagent-Live, NLP, Neuroscience Brain Encoding, and Radio Simulation domains are available in:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

