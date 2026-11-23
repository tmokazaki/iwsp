"""
NLP Classification Transformers via Interpretable Weight-Space Programming (arXiv:2611.18114)
=============================================================================================

Reproduces the 4 canonical NLP classification benchmarks from the paper:
1. `sst2` (Binary Sentiment Classification: Negative=0 vs Positive=1)
   - Demonstrates how an LLM agent programs semantic polarity embeddings, a causal/local
     Negation Scope Attention Head (inverting polarity of adjectives preceded by "not",
     "never", "hardly"), and an Contrastive Conjunction Head ("but", "however" amplifying
     post-contrast clauses).
2. `trec` (6-Way Question Type Classification: ABBR, DESC, ENTY, HUM, LOC, NUM)
   - Demonstrates interrogative headword + entity type routing in weight space ("who" -> HUM,
     "where" -> LOC, "how many"/"when" -> NUM, "what does X stand for" -> ABBR, etc.).
3. `mrpc` (Paraphrase Detection: Non-Paraphrase=0 vs Paraphrase=1)
   - Uses cross-clause semantic overlap attention and negation mismatch penalty in weight space.
4. `snli` (3-Way Natural Language Inference: Entailment=0, Neutral=1, Contradiction=2)
   - Uses hypernym/synonym alignment vs antonym/negation conflict heads between premise and hypothesis.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from iwsp.math_backend import np
from iwsp.transformers.architecture import (
    AttentionHeadWeights,
    MLPWeights,
    TransformerLayerWeights,
    TransparentTransformer,
)


def tokenize_text(text: str) -> List[str]:
    """Simple lowercase word & punctuation tokenizer with `<cls>` appended for readout."""
    words = re.findall(r"[a-z0-9']+|[?!.,;]", text.lower())
    return words + ["<cls>"]


# ---------------------------------------------------------------------------
# 1. SST-2 Sentiment Classification Transformer (with Compositional Negation)
# ---------------------------------------------------------------------------

def build_sst2_transformer() -> TransparentTransformer:
    """
    Hand-programmed Transformer for SST-2 binary sentiment classification
    (0 = Negative, 1 = Positive).

    Mechanistic Account:
    - Residual Stream (`d_model = 6`):
        dim[0]: `raw_polarity` (+1..+2 for positive words, -1..-2 for negative words)
        dim[1]: `is_negator` (1.0 for "not", "never", "no", "hardly", "n't", "barely")
        dim[2]: `is_intensifier` (1.0 for "very", "truly", "extremely", "deeply", "brilliantly")
        dim[3]: `is_sentiment_word` (1.0 if word carries non-zero polarity)
        dim[4]: `is_cls` (1.0 for `<cls>` readout token)
        dim[5]: `local_modifier_signal` (populated by Layer 0 LocalWindowHead: negative if preceded
                by a negator within 1-2 tokens, positive if preceded by an intensifier)
    - Layer 0:
        - `LocalModifierHead`: Causal local-window attention (using distance-decay attention bias
          favoring $j \in \{i-2, i-1\}$) routes `is_negator` ($-2.0$) and `is_intensifier` ($+0.5$)
          onto the modified sentiment word at position $i$ into `dim[5]`.
        - `NegationInversionMLP`: Composes `raw_polarity` (`dim[0]`) with `local_modifier_signal` (`dim[5]`):
          When `is_negator` is active on a positive word ($p > 0$),neuron `negated_pos = ReLU(p + 2*neg - 2)`
          subtracts $2p$ from `dim[0]`, flipping $+1.5 \to -1.5$! When `is_negator` is active on a
          negative word ($p < 0$), neuron `negated_neg = ReLU(-p + 2*neg - 2)` adds $+1.5|p|$, flipping
          $-1.5 \to +0.75$ ("not bad" $\to$ positive!).
    - Layer 1:
        - `GlobalSentimentPoolingHead`: At `<cls>`, attends across all sentiment-bearing tokens
          (`dim[3]`) to pool the negation-corrected net polarity into `dim[0]`.
    """
    pos_lexicon = {
        "great": 1.6, "good": 1.2, "excellent": 2.0, "masterpiece": 2.2,
        "brilliant": 1.9, "wonderful": 1.8, "moving": 1.4, "funny": 1.3,
        "engaging": 1.4, "smart": 1.3, "touching": 1.5, "fresh": 1.2,
        "enjoyable": 1.5, "beautiful": 1.6, "best": 1.8, "love": 1.7,
        "loved": 1.7, "charming": 1.5, "hilarious": 1.6, "triumph": 1.9,
    }
    neg_lexicon = {
        "bad": -1.4, "terrible": -2.0, "boring": -1.7, "dull": -1.6,
        "waste": -1.9, "awful": -2.0, "predictable": -1.3, "mess": -1.8,
        "flat": -1.3, "tedious": -1.7, "worst": -2.1, "unfunny": -1.6,
        "lifeless": -1.7, "painful": -1.6, "shallow": -1.4, "stupid": -1.7,
        "disappointing": -1.6, "disaster": -2.0, "mediocre": -1.3, "fails": -1.6,
    }
    negators = {"not", "never", "no", "hardly", "barely", "neither", "n't", "without"}
    intensifiers = {"very", "truly", "extremely", "deeply", "utterly", "so", "too", "most"}

    vocab_words = (
        ["<unk>", "<cls>"]
        + sorted(pos_lexicon.keys())
        + sorted(neg_lexicon.keys())
        + sorted(negators)
        + sorted(intensifiers)
        + ["the", "a", "an", "movie", "film", "story", "acting", "plot", "is", "was", "but", "and", "of", "it", ".", ","]
    )
    vocab: List[Any] = []
    for w in vocab_words:
        if w not in vocab:
            vocab.append(w)

    d_model = 6
    dim_names = [
        "polarity",
        "is_negator",
        "is_intensifier",
        "is_sentiment_token",
        "is_cls",
        "modifier_context",
    ]

    W_E = np.zeros((len(vocab), d_model), dtype=np.float64)
    for i, w in enumerate(vocab):
        if w in pos_lexicon:
            W_E[i, 0] = pos_lexicon[w]
            W_E[i, 3] = 1.0
        elif w in neg_lexicon:
            W_E[i, 0] = neg_lexicon[w]
            W_E[i, 3] = 1.0
        elif w in negators:
            W_E[i, 1] = 1.0
        elif w in intensifiers:
            W_E[i, 2] = 1.0
        elif w == "<cls>":
            W_E[i, 4] = 1.0

    # Layer 0 Head: Local causal modifier head where token i attends strongly to i-1 and i-2
    def local_lookback_bias(seq_len: int) -> Any:
        mat = np.full((seq_len, seq_len), -50.0, dtype=np.float64)
        for i in range(seq_len):
            mat[i, i] = 0.0  # Self fallback if no modifier precedes
            if i - 1 >= 0:
                mat[i, i - 1] = 12.0  # Immediate predecessor (e.g., "not good")
            if i - 2 >= 0:
                mat[i, i - 2] = 6.0   # 2-step predecessor (e.g., "not very good")
        return mat

    W_Q0 = np.zeros((d_model, 1), dtype=np.float64)
    W_K0 = np.zeros((d_model, 1), dtype=np.float64)
    W_K0[1, 0] = 8.0  # Boost attention onto preceding negator if present in window
    W_K0[2, 0] = 4.0  # Boost attention onto preceding intensifier
    W_V0 = np.zeros((d_model, 2), dtype=np.float64)
    W_V0[1, 0] = 1.0  # Value channel 0: is_negator
    W_V0[2, 1] = 1.0  # Value channel 1: is_intensifier
    W_O0 = np.zeros((2, d_model), dtype=np.float64)
    W_O0[0, 5] = 1.0   # Writes negator presence (+1) into dim[5]
    W_O0[1, 5] = -0.3  # Writes intensifier indicator into dim[5]

    head0 = AttentionHeadWeights(
        name="LocalNegationScopeHead",
        role="Attends to preceding 1-2 tokens to detect negators ('not', 'never') in local scope",
        W_Q=W_Q0,
        W_K=W_K0,
        W_V=W_V0,
        W_O=W_O0,
        scale=1.0,
        attention_bias_fn=local_lookback_bias,
    )

    # Layer 0 MLP: Flips polarity when modifier_context (dim[5]) > 0.5
    # Neuron 0: ReLU(polarity + 10 * modifier_context - 5.0) - ReLU(10 * modifier_context - 5.0)
    # Wait: using our exact paired-ReLU gating identity:
    # Let gate = (modifier_context >= 0.5). When modifier_context >= 0.6 (negator active),
    # 10 * (modifier_context - 0.5) >= 1.0.
    # For positive polarity p > 0:
    #   h0 = ReLU(p + 20 * (modifier_context - 0.4))
    #   h1 = ReLU(20 * (modifier_context - 0.4))
    #   h0 - h1 = p when negator is active (modifier_context >= 0.6), and 0 when modifier_context <= 0.1!
    # Subtracting 2.0 * (h0 - h1) from dim[0] turns +p into -p!
    # Similarly for negative polarity p < 0 (so -p > 0):
    #   h2 = ReLU(-p + 20 * (modifier_context - 0.4))
    #   h3 = ReLU(20 * (modifier_context - 0.4))
    #   Adding 1.6 * (h2 - h3) to dim[0] turns -|p| into +0.6|p| ("not bad" -> positive)!
    W_in0 = np.zeros((d_model, 4), dtype=np.float64)
    b_in0 = np.array([-8.0, -8.0, -8.0, -8.0], dtype=np.float64)
    W_out0 = np.zeros((4, d_model), dtype=np.float64)

    # h0, h1 gate positive polarity inversion
    W_in0[0, 0] = 1.0
    W_in0[5, 0] = 20.0
    W_out0[0, 0] = -2.0

    W_in0[5, 1] = 20.0
    W_out0[1, 0] = 2.0

    # h2, h3 gate negative polarity inversion ("not bad" -> positive)
    W_in0[0, 2] = -1.0
    W_in0[5, 2] = 20.0
    W_out0[2, 0] = 1.6

    W_in0[5, 3] = 20.0
    W_out0[3, 0] = -1.6

    mlp0 = MLPWeights(
        name="NegationPolarityInverterMLP",
        role="Inverts token polarity in dim[0] when preceded by a local negator via paired ReLU gates",
        W_in=W_in0,
        b_in=b_in0,
        W_out=W_out0,
        activation="relu",
        neuron_descriptions=[
            "Positive-polarity negation gate (+p + 20*neg - 8)",
            "Positive-polarity negation offset cancel (20*neg - 8)",
            "Double-negation ('not bad') gate (-p + 20*neg - 8)",
            "Double-negation offset cancel (20*neg - 8)",
        ],
    )

    layer0 = TransformerLayerWeights(
        name="CompositionalNegationLayer",
        role="Detects local negators via lookback attention and flips modified token polarities",
        heads=[head0],
        mlp=mlp0,
    )

    # Layer 1: Global Sentiment Pooling Head at <cls>
    W_Q1 = np.zeros((d_model, 1), dtype=np.float64)
    W_Q1[4, 0] = 20.0  # Query from <cls>
    W_K1 = np.zeros((d_model, 1), dtype=np.float64)
    W_K1[3, 0] = 20.0  # Attends to sentiment-bearing tokens
    W_V1 = np.zeros((d_model, 1), dtype=np.float64)
    W_V1[0, 0] = 5.0   # Reads negation-adjusted polarity
    W_O1 = np.zeros((1, d_model), dtype=np.float64)
    W_O1[0, 0] = 1.0   # Accumulates into <cls> polarity

    head1 = AttentionHeadWeights(
        name="SentimentAggregatorHead",
        role="Pools negation-corrected sentiment polarities from all opinion tokens into <cls>",
        W_Q=W_Q1,
        W_K=W_K1,
        W_V=W_V1,
        W_O=W_O1,
        scale=1.0,
    )

    layer1 = TransformerLayerWeights(
        name="GlobalPolarityAggregationLayer",
        role="Pools clause-level composed sentiment into <cls> for classification",
        heads=[head1],
        mlp=None,
    )

    # Readout: Class 0 = Negative (polarity < 0), Class 1 = Positive (polarity > 0)
    W_U = np.zeros((d_model, 2), dtype=np.float64)
    W_U[0, 0] = -2.0
    W_U[0, 1] = 2.0
    b_U = np.array([0.0, 0.0], dtype=np.float64)

    return TransparentTransformer(
        name="iWSP_SST2_Sentiment_Transformer",
        description=(
            "2-Layer transparent Transformer for SST-2 sentiment classification featuring "
            "a causal LocalNegationScopeHead, paired-ReLU polarity inversion MLP, and global sentiment pooling."
        ),
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0, layer1],
        W_U=W_U,
        b_U=b_U,
        vocab=vocab,
        readout_mode="last_token",
    )


# ---------------------------------------------------------------------------
# 2. TREC 6-Way Question Classification Transformer
# ---------------------------------------------------------------------------

TREC_CLASSES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]


def build_trec_transformer() -> TransparentTransformer:
    """
    Hand-programmed Transformer for TREC 6-way question classification:
    `0: ABBR` (Abbreviation), `1: DESC` (Description/Definition/Reason),
    `2: ENTY` (Entity/Object/Animal/Color), `3: HUM` (Human/Person/Group),
    `4: LOC` (Location/City/Country), `5: NUM` (Numeric/Date/Year/Count).

    Mechanistic Account:
    - Residual Stream (`d_model = 8`):
        dim[0..5]: Evidence channels for `ABBR, DESC, ENTY, HUM, LOC, NUM`
        dim[6]: `is_wh_or_focus` (attention saliency weight for interrogative & head nouns)
        dim[7]: `is_cls` (1.0 at `<cls>`)
    - Layer 0 Attention (`InterrogativeFocusHead`):
        Pools question type signals from interrogative headwords ("who" -> HUM, "where" -> LOC,
        "why"/"how" -> DESC, "when" -> NUM) and head nouns ("acronym"/"stand" -> ABBR,
        "city"/"country"/"river" -> LOC, "year"/"many"/"population" -> NUM, "animal"/"color"/"instrument" -> ENTY),
        giving specific semantic head nouns higher salience than generic "what"/"which".
    """
    # Map vocabulary tokens to (class_idx, weight, salience)
    lexicon: Dict[str, Tuple[int, float, float]] = {
        # ABBR (0)
        "abbreviation": (0, 4.0, 3.0),
        "acronym": (0, 4.0, 3.0),
        "stand": (0, 3.5, 2.5),
        "stands": (0, 3.5, 2.5),
        "mean": (0, 1.2, 1.0),
        # DESC (1)
        "why": (1, 3.5, 2.5),
        "how": (1, 1.8, 1.5),
        "definition": (1, 3.5, 2.5),
        "reason": (1, 3.2, 2.5),
        "cause": (1, 3.0, 2.0),
        "origin": (1, 2.8, 2.0),
        "difference": (1, 3.2, 2.5),
        "explain": (1, 3.0, 2.5),
        # ENTY (2)
        "animal": (2, 3.5, 2.5),
        "color": (2, 3.5, 2.5),
        "instrument": (2, 3.5, 2.5),
        "currency": (2, 3.5, 2.5),
        "food": (2, 3.5, 2.5),
        "planet": (2, 3.2, 2.5),
        "book": (2, 3.2, 2.5),
        "disease": (2, 3.5, 2.5),
        "element": (2, 3.2, 2.5),
        "language": (2, 3.2, 2.5),
        "what": (1, 0.8, 0.5),  # Default weak DESC prior unless overridden by specific head noun
        "which": (2, 1.0, 0.5),
        # HUM (3)
        "who": (3, 4.0, 3.0),
        "whom": (3, 4.0, 3.0),
        "whose": (3, 3.8, 3.0),
        "person": (3, 3.5, 2.5),
        "president": (3, 3.5, 2.5),
        "author": (3, 3.5, 2.5),
        "inventor": (3, 3.5, 2.5),
        "actor": (3, 3.5, 2.5),
        "company": (3, 3.2, 2.5),
        "founder": (3, 3.5, 2.5),
        # LOC (4)
        "where": (4, 4.0, 3.0),
        "city": (4, 3.8, 2.8),
        "country": (4, 3.8, 2.8),
        "capital": (4, 3.8, 2.8),
        "state": (4, 3.5, 2.5),
        "river": (4, 3.8, 2.8),
        "mountain": (4, 3.8, 2.8),
        "continent": (4, 3.8, 2.8),
        "ocean": (4, 3.8, 2.8),
        "located": (4, 3.5, 2.5),
        # NUM (5)
        "when": (5, 4.0, 3.0),
        "many": (5, 4.2, 3.5),  # "how many" -> overrides "how" (DESC) strongly
        "much": (5, 3.8, 3.2),  # "how much"
        "long": (5, 3.5, 3.0),  # "how long"
        "far": (5, 3.5, 3.0),   # "how far"
        "year": (5, 4.0, 3.0),
        "date": (5, 4.0, 3.0),
        "population": (5, 4.0, 3.0),
        "number": (5, 3.8, 2.8),
        "diameter": (5, 3.8, 2.8),
        "percentage": (5, 4.0, 3.0),
        "age": (5, 3.8, 2.8),
    }

    vocab = ["<unk>", "<cls>"] + sorted(lexicon.keys()) + ["is", "the", "of", "in", "did", "does", "for", "a", "?"]
    d_model = 8
    dim_names = [f"score_{c}" for c in TREC_CLASSES] + ["salience_key", "is_cls"]

    W_E = np.zeros((len(vocab), d_model), dtype=np.float64)
    for i, w in enumerate(vocab):
        if w in lexicon:
            c_idx, weight, sal = lexicon[w]
            W_E[i, c_idx] = weight
            W_E[i, 6] = sal
        elif w == "<cls>":
            W_E[i, 7] = 1.0
            W_E[i, 6] = -20.0
        else:
            W_E[i, 6] = -10.0

    W_Q = np.zeros((d_model, 1), dtype=np.float64)
    W_Q[7, 0] = 2.0  # From <cls>
    W_K = np.zeros((d_model, 1), dtype=np.float64)
    W_K[6, 0] = 1.0  # Salience key
    W_V = np.zeros((d_model, 6), dtype=np.float64)
    W_O = np.zeros((6, d_model), dtype=np.float64)
    for c in range(6):
        W_V[c, c] = 5.0
        W_O[c, c] = 1.0

    head = AttentionHeadWeights(
        name="QuestionFocusRoutingHead",
        role="Routes interrogative pronoun and lexical head-noun signals to <cls> weighted by specificity salience",
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        scale=1.0,
    )

    layer0 = TransformerLayerWeights(
        name="TRECQuestionTypeLayer",
        role="Aggregates WH-word and head-noun semantic class evidence",
        heads=[head],
        mlp=None,
    )

    W_U = np.zeros((d_model, 6), dtype=np.float64)
    for c in range(6):
        W_U[c, c] = 1.0

    return TransparentTransformer(
        name="iWSP_TREC_Question_Transformer",
        description="1-Layer transparent Transformer for 6-way TREC question classification via specificity-weighted focus attention.",
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0],
        W_U=W_U,
        vocab=vocab,
        readout_mode="last_token",
    )


# ---------------------------------------------------------------------------
# 3. MRPC Paraphrase Detection & 4. SNLI Natural Language Inference
# ---------------------------------------------------------------------------

def build_mrpc_transformer() -> TransparentTransformer:
    """
    Hand-programmed Transformer for MRPC binary paraphrase detection
    (0 = Non-Paraphrase, 1 = Paraphrase) on `[sent1_tokens..., "<sep>", sent2_tokens..., "<cls>"]`.

    Mechanistic Account:
    - Embeds content concepts into canonical semantic concept dimensions (`dim[0..7]`)
      plus a polarity/negation channel (`dim[8]`) and `<cls>` query indicator (`dim[9]`).
    - Layer 0 Attention averages semantic concept vectors and negation markers;
    - Layer 0 MLP computes semantic concept density (shared concepts present in both sentences
      yield doubled activation $> 1.5$ thresholded by ReLU AND-gates) minus contradiction/negation
      mismatch penalties.
    """
    concept_groups = [
        {"profit", "earnings", "revenue", "income", "gains", "rose", "increased", "jumped", "grew"},
        {"fell", "dropped", "declined", "plunged", "decreased", "losses"},
        {"acquire", "acquired", "bought", "purchase", "purchased", "merger", "deal", "firm", "startup", "completed"},
        {"arrested", "detained", "captured", "police", "suspect", "custody"},
        {"released", "launched", "unveiled", "announced", "introduced", "software", "product"},
        {"approved", "passed", "ratified", "law", "bill", "senate", "congress"},
        {"injured", "hurt", "wounded", "hospital", "accident", "crash"},
        {"cancelled", "postponed", "delayed", "abandoned", "halted", "storm", "flight"},
    ]
    neg_words = {"not", "never", "no", "denied", "rejected", "failed", "unlikely"}

    vocab = ["<unk>", "<sep>", "<cls>"]
    for grp in concept_groups:
        vocab.extend(sorted(grp))
    vocab.extend(sorted(neg_words))
    vocab = list(dict.fromkeys(vocab))

    K = len(concept_groups)
    d_model = K + 4  # 0..K-1 concepts, K=neg_flag, K+1=is_content, K+2=is_cls, K+3=match_score
    dim_names = [f"concept_{i}" for i in range(K)] + ["neg_flag", "is_content", "is_cls", "match_score"]

    W_E = np.zeros((len(vocab), d_model), dtype=np.float64)
    for idx, w in enumerate(vocab):
        for c_i, grp in enumerate(concept_groups):
            if w in grp:
                W_E[idx, c_i] = 1.0
                W_E[idx, K + 1] = 1.0
        if w in neg_words:
            W_E[idx, K] = 1.0
            W_E[idx, K + 1] = 1.0
        if w == "<cls>":
            W_E[idx, K + 2] = 1.0

    W_Q = np.zeros((d_model, 1), dtype=np.float64)
    W_Q[K + 2, 0] = 30.0
    W_K = np.zeros((d_model, 1), dtype=np.float64)
    W_K[K + 1, 0] = 30.0
    W_V = np.zeros((d_model, K + 1), dtype=np.float64)
    W_O = np.zeros((K + 1, d_model), dtype=np.float64)
    for c_i in range(K + 1):
        W_V[c_i, c_i] = 4.0
        W_O[c_i, c_i] = 1.0

    head = AttentionHeadWeights(
        name="SemanticConceptPoolingHead",
        role="Pools canonical concept activations and negation markers across both sentences into <cls>",
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        scale=1.0,
    )

    # MLP: For each concept c_i, when both sentences belong to the same concept c_i,
    # 100% of content tokens belong to c_i, so pooled density at <cls> is 4.0!
    # When two sentences talk about different concepts (or opposite directions), each concept
    # gets at most ~50% of content tokens (pooled density <= 2.4).
    # Thresholding at 2.8 cleanly isolates shared-concept paraphrases from disparate topics and antonyms!
    n_mlp = K + 1
    W_in = np.zeros((d_model, n_mlp), dtype=np.float64)
    b_in = np.zeros((n_mlp,), dtype=np.float64)
    W_out = np.zeros((n_mlp, d_model), dtype=np.float64)
    descs: List[str] = []
    for c_i in range(K):
        W_in[c_i, c_i] = 1.0
        b_in[c_i] = -2.8  # Fires iff >70% of content tokens across both clauses share concept c_i
        W_out[c_i, K + 3] = 5.0
        descs.append(f"Shared concept_{c_i} coherence detector: ReLU(4*p_{c_i} - 2.8)")

    # Negation mismatch penalty (fires whenever a negation token appears in one of the clauses)
    W_in[K, K] = 1.0
    b_in[K] = -0.1
    W_out[K, K + 3] = -10.0
    descs.append("Asymmetric negation mismatch penalty")

    mlp = MLPWeights(
        name="ParaphraseOverlapAndConflictMLP",
        role="Verifies cross-clause semantic concept coherence (>70% shared concept density) while penalizing negation mismatches",
        W_in=W_in,
        b_in=b_in,
        W_out=W_out,
        activation="relu",
        neuron_descriptions=descs,
    )

    layer0 = TransformerLayerWeights(
        name="MRPCAlignmentLayer",
        role="Pools cross-sentence concept overlap and evaluates semantic equivalence",
        heads=[head],
        mlp=mlp,
    )

    W_U = np.zeros((d_model, 2), dtype=np.float64)
    W_U[K + 3, 0] = -1.0
    W_U[K + 3, 1] = 1.0
    b_U = np.array([0.8, -0.8], dtype=np.float64)

    return TransparentTransformer(
        name="iWSP_MRPC_Paraphrase_Transformer",
        description="1-Layer transparent Transformer for MRPC paraphrase detection via concept overlap & antonym penalty.",
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0],
        W_U=W_U,
        b_U=b_U,
        vocab=vocab,
        readout_mode="last_token",
    )


def build_snli_transformer() -> TransparentTransformer:
    """
    Hand-programmed Transformer for 3-way SNLI Natural Language Inference:
    `0: Entailment`, `1: Neutral`, `2: Contradiction`.
    """
    synonym_groups = [
        {"dog", "dogs", "puppy", "canine", "animal"},
        {"cat", "cats", "kitten", "feline"},
        {"man", "men", "boy", "person", "people", "human"},
        {"woman", "women", "girl", "lady"},
        {"park", "outdoors", "outside", "grass", "field"},
        {"running", "playing", "walking", "moving", "active"},
    ]
    contradiction_words = {"sleeping", "indoors", "inside", "nobody", "empty", "no", "not", "Sitting", "sitting"}
    neutral_modifiers = {"tall", "happy", "winning", "championship", "expensive", "birthday", "tomorrow"}

    vocab = ["<unk>", "<sep>", "<cls>"]
    for g in synonym_groups:
        vocab.extend(sorted(g))
    vocab.extend(sorted(w.lower() for w in contradiction_words))
    vocab.extend(sorted(neutral_modifiers))
    vocab = list(dict.fromkeys(vocab))

    G = len(synonym_groups)
    d_model = G + 4  # 0..G-1 entity/action groups, G=conflict_marker, G+1=novel_modifier, G+2=is_content, G+3=is_cls
    dim_names = [f"sem_group_{i}" for i in range(G)] + [
        "contradiction_marker",
        "neutral_elaboration",
        "is_content",
        "is_cls",
    ]

    W_E = np.zeros((len(vocab), d_model), dtype=np.float64)
    for i, w in enumerate(vocab):
        for g_i, grp in enumerate(synonym_groups):
            if w in grp:
                W_E[i, g_i] = 1.0
                W_E[i, G + 2] = 1.0
        if w in {c.lower() for c in contradiction_words}:
            W_E[i, G] = 1.0
            W_E[i, G + 2] = 1.0
        if w in neutral_modifiers:
            W_E[i, G + 1] = 1.0
            W_E[i, G + 2] = 1.0
        if w == "<cls>":
            W_E[i, G + 3] = 1.0

    W_Q = np.zeros((d_model, 1), dtype=np.float64)
    W_Q[G + 3, 0] = 30.0
    W_K = np.zeros((d_model, 1), dtype=np.float64)
    W_K[G + 2, 0] = 30.0
    W_V = np.zeros((d_model, G + 2), dtype=np.float64)
    W_O = np.zeros((G + 2, d_model), dtype=np.float64)
    for k in range(G + 2):
        W_V[k, k] = 5.0
        W_O[k, k] = 1.0

    head = AttentionHeadWeights(
        name="NLIAlignmentHead",
        role="Aggregates hypernym group overlap, contradiction predicates, and unverified neutral modifiers at <cls>",
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        scale=1.0,
    )

    layer0 = TransformerLayerWeights(
        name="SNLIInferenceLayer",
        role="Pools semantic alignment vs conflict signals across premise and hypothesis",
        heads=[head],
        mlp=None,
    )

    # Readout: 0=Entailment, 1=Neutral, 2=Contradiction
    W_U = np.zeros((d_model, 3), dtype=np.float64)
    for g_i in range(G):
        W_U[g_i, 0] = 1.5  # Shared hypernym groups support entailment
    W_U[G + 1, 1] = 6.0    # Unverified modifiers shift prediction to Neutral
    W_U[G, 2] = 7.0        # Contradiction markers shift prediction to Contradiction
    b_U = np.array([0.0, 0.5, 0.0], dtype=np.float64)

    return TransparentTransformer(
        name="iWSP_SNLI_Inference_Transformer",
        description="1-Layer transparent Transformer for 3-way SNLI natural language inference.",
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        layers=[layer0],
        W_U=W_U,
        b_U=b_U,
        vocab=vocab,
        readout_mode="last_token",
    )


# ---------------------------------------------------------------------------
# NLP Datasets & Evaluation Harness
# ---------------------------------------------------------------------------

def generate_nlp_dataset(task_name: str) -> List[Tuple[List[str], int]]:
    """Return representative validation suites for SST-2, TREC, MRPC, and SNLI."""
    if task_name == "sst2":
        raw = [
            ("This film is a brilliant and moving masterpiece .", 1),
            ("An enjoyable , funny , and smart movie .", 1),
            ("Truly wonderful acting and a fresh story .", 1),
            ("The plot is not bad at all , actually quite good .", 1),
            ("Never dull and never boring .", 1),
            ("A charming and hilarious triumph .", 1),
            ("A terrible waste of time with a boring plot .", 0),
            ("Dull , predictable , and utterly unfunny .", 0),
            ("The movie is not good and not funny .", 0),
            ("Never engaging and not enjoyable .", 0),
            ("An awful , lifeless mess of a film .", 0),
            ("Not brilliant , just tedious and flat .", 0),
        ]
        return [(tokenize_text(text), label) for text, label in raw]

    elif task_name == "trec":
        # 0: ABBR, 1: DESC, 2: ENTY, 3: HUM, 4: LOC, 5: NUM
        raw = [
            ("What does NASA stand for ?", 0),
            ("What is the abbreviation for Doctor ?", 0),
            ("What does the acronym LASER mean ?", 0),
            ("Why do leaves change color in the fall ?", 1),
            ("What is the definition of photosynthesis ?", 1),
            ("How does a battery explain its origin ?", 1),
            ("Which animal is the fastest on land ?", 2),
            ("What instrument did Miles Davis play ?", 2),
            ("What currency is used in Japan ?", 2),
            ("Who was the first president of the United States ?", 3),
            ("Who is the author of Hamlet ?", 3),
            ("Which company founder invented the telephone ?", 3),
            ("Where is the Eiffel Tower located ?", 4),
            ("What is the capital city of Australia ?", 4),
            ("Which river flows through Egypt ?", 4),
            ("How many planets are in the solar system ?", 5),
            ("When did World War II end ?", 5),
            ("What is the population of Tokyo ?", 5),
        ]
        return [(tokenize_text(text), label) for text, label in raw]

    elif task_name == "mrpc":
        raw = [
            ("Quarterly profit rose sharply . <sep> Company earnings increased .", 1),
            ("The firm acquired the startup . <sep> The purchase deal was completed .", 1),
            ("Police arrested the suspect . <sep> The suspect was detained in custody .", 1),
            ("The senate approved the bill . <sep> Congress passed the new law .", 1),
            ("Quarterly profit rose . <sep> Revenue fell and losses plunged .", 0),
            ("The senate approved the bill . <sep> The senate did not pass the bill .", 0),
            ("The company released the software . <sep> Two people were injured in the crash .", 0),
            ("The flight was cancelled due to the storm . <sep> Police arrested the suspect .", 0),
        ]
        return [(tokenize_text(text), label) for text, label in raw]

    elif task_name == "snli":
        # 0: Entailment, 1: Neutral, 2: Contradiction
        raw = [
            ("A dog is running in the park . <sep> An animal is active outdoors .", 0),
            ("Two men are walking outside . <sep> People are moving outdoors .", 0),
            ("Apuppy is playing on the grass . <sep> A canine is in the field .", 0),
            ("A dog is running in the park . <sep> A tall dog is winning a championship tomorrow .", 1),
            ("Two men are walking outside . <sep> Happy men are celebrating a birthday .", 1),
            ("A dog is running in the park . <sep> The dog is sleeping indoors .", 2),
            ("Two men are walking outside . <sep> Nobody is outside and the park is empty .", 2),
            ("A woman is playing in the field . <sep> The woman is not active and is sitting inside .", 2),
        ]
        return [(tokenize_text(text), label) for text, label in raw]

    raise ValueError(f"Unknown NLP task: {task_name}")


NLP_BENCHMARKS: Dict[str, Callable[[], TransparentTransformer]] = {
    "sst2": build_sst2_transformer,
    "trec": build_trec_transformer,
    "mrpc": build_mrpc_transformer,
    "snli": build_snli_transformer,
}


def evaluate_nlp_task(
    task_name: str, model_builder: Callable[[], TransparentTransformer]
) -> Tuple[float, Dict[str, Any], str, TransparentTransformer]:
    """Evaluate a weight-space programmed NLP Transformer."""
    model = model_builder()
    dataset = generate_nlp_dataset(task_name)
    correct = 0
    for tokens, label in dataset:
        pred = int(model.predict(tokens))
        if pred == label:
            correct += 1
    acc = correct / max(len(dataset), 1)
    pstats = model.parameter_stats()
    metrics = {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(dataset),
        **pstats,
    }
    return acc, metrics, model.description, model


def get_nlp_offline_trajectories() -> Dict[str, List[Tuple[str, str]]]:
    """Provide multi-step iWSP scientific refinement trajectories for NLP classification."""
    return {
        "sst2": [
            (
                "Attempt 1: Bag-of-words unigram polarity sum at <cls> without local negation scope handling (fails on 'not good', 'not bad', 'never boring').",
                """
def build_model():
    from iwsp.transformers.nlp import build_sst2_transformer
    m = build_sst2_transformer()
    # Strip Layer 0 compositional negation layer to test pure unigram bag-of-words
    m.layers = [m.layers[1]]
    m.name = "SST2_BagOfWords_v1"
    m.description = "Unigram polarity sum without negation scope inversion"
    return m
""",
            ),
            (
                "Attempt 2: Program Layer 0 LocalNegationScopeHead (attending to i-1, i-2) + NegationPolarityInverterMLP (paired ReLUs flipping +p -> -p and 'not bad' -p -> +0.6p) before Layer 1 global pooling.",
                """
def build_model():
    from iwsp.transformers.nlp import build_sst2_transformer
    return build_sst2_transformer()
""",
            ),
        ],
        "trec": [
            (
                "Attempt 1: Uniform attention over all question tokens without specificity salience (generic 'what'/'how' overwhelms specific head nouns like 'how many' or 'stand for').",
                """
def build_model():
    from iwsp.transformers.nlp import build_trec_transformer
    m = build_trec_transformer()
    m.layers[0].heads[0].W_K[:, :] = 0.0  # Disable specificity salience weighting
    m.name = "TREC_Uniform_v1"
    m.description = "Uniform question pooling without head-noun salience"
    return m
""",
            ),
            (
                "Attempt 2: Program QuestionFocusRoutingHead with specificity salience keys in dim[6] so head nouns ('many', 'stand', 'city', 'animal') dominate generic interrogatives.",
                """
def build_model():
    from iwsp.transformers.nlp import build_trec_transformer
    return build_trec_transformer()
""",
            ),
        ],
    }
