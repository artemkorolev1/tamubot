from __future__ import annotations

import random

import pandas as pd

from tamubot.ingestion.pipeline_v6b.util.text_normalize import (
    ReferenceIndex,
    jaccard,
    minhash_of,
    ngrams,
    normalize_text,
    stable_cluster_id,
)


def test_normalize_text_is_idempotent():
    raw = "  Hello,   WORLD!\n\tThis is a TEST.  "
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice


def test_normalize_text_lowercases_and_collapses_whitespace():
    out = normalize_text("Hello\t\tWORLD\n  foo")
    assert out == "hello world foo"


def test_normalize_text_strips_punctuation():
    out = normalize_text("Don't stop! It's: a-test, OK?")
    assert "'" not in out
    assert "!" not in out
    assert "," not in out
    assert ":" not in out
    assert "-" not in out
    assert "?" not in out


def test_ngrams_returns_empty_when_fewer_than_n_tokens():
    assert ngrams("one two three four", n=5) == []
    assert ngrams("", n=5) == []


def test_ngrams_count_is_len_tokens_minus_n_plus_one():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    grams = ngrams(text, n=5)
    assert len(grams) == 8 - 5 + 1
    assert grams[0] == "alpha beta gamma delta epsilon"
    assert grams[-1] == "delta epsilon zeta eta theta"


def test_jaccard_zero_for_disjoint_sets():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_one_for_identical_sets():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_handles_two_empty_sets():
    assert jaccard(set(), set()) == 0.0


def test_minhash_of_distinct_short_texts_do_not_collide():
    # Short (<5-token) texts must NOT collapse to the empty MinHash (which has
    # Jaccard 1.0 with everything) — distinct short strings get distinct sigs.
    a = minhash_of("office hours")
    b = minhash_of("grading policy")
    assert a.jaccard(b) < 0.5


def test_minhash_of_identical_short_texts_still_match():
    a = minhash_of("office hours")
    b = minhash_of("Office   Hours!")  # identical after normalization
    assert a.jaccard(b) == 1.0


def test_minhash_of_truly_empty_text_is_empty_minhash():
    # Punctuation/whitespace-only normalizes to "" → empty MinHash. Callers skip
    # these (build_local_signatures / scan filter on normalize_text), so the
    # empty-vs-empty 1.0 here never reaches a dedup decision.
    assert minhash_of("").jaccard(minhash_of("  !!!  ")) == 1.0


def test_minhash_jaccard_approximates_true_jaccard():
    rng = random.Random(42)
    vocab = [f"tok{i}" for i in range(500)]
    base_tokens = [rng.choice(vocab) for _ in range(200)]
    perturbed = list(base_tokens)
    # Replace 5% of tokens with fresh draws.
    n_replace = max(1, int(0.05 * len(perturbed)))
    for idx in rng.sample(range(len(perturbed)), n_replace):
        perturbed[idx] = f"new{idx}"

    text_a = " ".join(base_tokens)
    text_b = " ".join(perturbed)

    grams_a = set(ngrams(text_a, n=5))
    grams_b = set(ngrams(text_b, n=5))
    true_jacc = jaccard(grams_a, grams_b)

    mh_a = minhash_of(text_a)
    mh_b = minhash_of(text_b)
    est = mh_a.jaccard(mh_b)

    assert abs(est - true_jacc) < 0.10


def test_stable_cluster_id_deterministic():
    a1 = stable_cluster_id("hello world")
    a2 = stable_cluster_id("hello world")
    b = stable_cluster_id("hello worlds")
    assert a1 == a2
    assert a1 != b
    assert a1.startswith("bp_")
    assert len(a1) == len("bp_") + 16


def _write_tiny_reference_parquet(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "ref.parquet"
    df.to_parquet(path)
    return path


def test_reference_index_empty_match_returns_none():
    idx = ReferenceIndex.empty()
    cid, conf = idx.match("anything at all goes here")
    assert cid is None
    assert conf is None


def test_reference_index_exact_match_returns_confidence_1(tmp_path):
    rep = (
        "Texas A and M University students must adhere to the Aggie Honor Code "
        "which states that an Aggie does not lie cheat or steal nor tolerate "
        "those who do throughout all academic activities"
    )
    norm = normalize_text(rep)
    cid = stable_cluster_id(norm)
    path = _write_tiny_reference_parquet(
        tmp_path,
        [
            {
                "cluster_id": cid,
                "representative_text": rep,
                "normalized_text": norm,
                "doc_frequency": 7,
                "distinct_depts": 4,
            }
        ],
    )
    idx = ReferenceIndex(path)
    got_id, conf = idx.match(rep)
    assert got_id == cid
    assert conf == 1.0


def test_reference_index_fuzzy_match_above_threshold(tmp_path):
    base_tokens = [f"word{i}" for i in range(100)]
    rep = " ".join(base_tokens)
    norm = normalize_text(rep)
    cid = stable_cluster_id(norm)
    path = _write_tiny_reference_parquet(
        tmp_path,
        [
            {
                "cluster_id": cid,
                "representative_text": rep,
                "normalized_text": norm,
                "doc_frequency": 6,
                "distinct_depts": 3,
            }
        ],
    )
    idx = ReferenceIndex(path)

    # Swap 1 token of 100 — true Jaccard ~0.90, comfortably above the 0.80 LSH band.
    perturbed_tokens = list(base_tokens)
    perturbed_tokens[37] = "swapped"
    perturbed = " ".join(perturbed_tokens)

    got_id, conf = idx.match(perturbed)
    assert got_id == cid
    assert conf is not None
    assert conf >= 0.80


def test_reference_index_no_match_below_threshold(tmp_path):
    rep = (
        "Texas A and M University students must adhere to the Aggie Honor Code "
        "which states that an Aggie does not lie cheat or steal nor tolerate "
        "those who do throughout all academic activities"
    )
    norm = normalize_text(rep)
    cid = stable_cluster_id(norm)
    path = _write_tiny_reference_parquet(
        tmp_path,
        [
            {
                "cluster_id": cid,
                "representative_text": rep,
                "normalized_text": norm,
                "doc_frequency": 5,
                "distinct_depts": 3,
            }
        ],
    )
    idx = ReferenceIndex(path)

    unrelated = (
        "The quantum harmonic oscillator is a fundamental model in physics "
        "describing the motion of a particle in a parabolic potential well "
        "with discrete energy eigenstates spaced by Planck constant times omega"
    )
    got_id, conf = idx.match(unrelated)
    assert got_id is None
    assert conf is None
