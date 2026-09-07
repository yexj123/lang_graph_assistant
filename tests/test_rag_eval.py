"""RAG quality evaluation against a sourced golden set.

Deselected by default (`deepeval` marker) because it needs a live Postgres with an
ingested corpus and makes billed API calls - roughly one judge call per metric per
golden. Run it deliberately:

    python fetch_papers.py && python ingest.py     # build the corpus and index
    pytest -m deepeval                             # the retrieval arm
    pytest -m baseline                             # the no-retrieval comparison

What this fixes relative to the original two-item smoke test:

* The judge is a different model from the generator. Grading your own output has a
  documented self-preference bias, and it inflates faithfulness and relevancy - the
  two metrics that carry the most weight here.
* Thresholds are declared below instead of silently inheriting deepeval's 0.5.
* The golden set is sourced: every reference answer names the paper and page it came
  from, and tests/test_golden_set.py verifies the quote is really there.
* There is a no-retrieval baseline, so a good score can be attributed to retrieval
  rather than to what the generator already knew.

Every import of config is deferred: pytest imports this module during collection to
read its markers, and config reaches Postgres on first use.
"""

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GOLDENS = json.loads((Path(__file__).parent / "golden_set.json").read_text(encoding="utf-8"))["goldens"]
_IDS = [g["source"] + f"-p{g['page']}" for g in GOLDENS]

# Declared, not inherited. deepeval defaults every metric to 0.5; these are the bars
# this project actually commits to, and they differ because the stakes differ.
FAITHFULNESS_THRESHOLD = 0.8  # a thesis tool asserting unsupported claims is the worst failure
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXTUAL_PRECISION_THRESHOLD = 0.6  # top-k will always carry some irrelevant chunks
CONTEXTUAL_RECALL_THRESHOLD = 0.6
CORRECTNESS_THRESHOLD = 0.7

RETRIEVAL_K = 4

_EMPTY_INDEX_HINT = (
    "The literature index returned nothing. Run `python fetch_papers.py` then "
    "`python ingest.py` before evaluating, otherwise these scores measure an empty corpus."
)


def _judge_model() -> str:
    """Fail loudly rather than silently self-grading."""
    from config import JUDGE_MODEL_NAME, MODEL_NAME

    if JUDGE_MODEL_NAME == MODEL_NAME:
        pytest.fail(
            f"Judge and generator are both '{MODEL_NAME}'. Self-evaluation inflates these "
            f"scores; set JUDGE_MODEL to a different model before trusting the results."
        )
    return JUDGE_MODEL_NAME


def _correctness_metric(judge: str) -> GEval:
    """Does the answer actually say what the sourced reference says?

    The four RAG metrics score grounding and relevance, none of them correctness against
    a known-good answer - which is the only thing comparable across the two arms.
    """
    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output conveys the same factual content as the "
            "expected output. Numbers, model names and quantities must match. Additional "
            "correct detail is acceptable; contradicting or missing the key fact is not."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=CORRECTNESS_THRESHOLD,
    )


def _retrieve(query: str) -> list[str]:
    from config import get_vector_store

    docs = get_vector_store().similarity_search(query, k=RETRIEVAL_K)
    return [doc.page_content for doc in docs]


def _answer(query: str, retrieval_context: list[str]) -> str:
    from config import get_model

    if retrieval_context:
        prompt = (
            "Context from literature:\n" + "\n\n".join(retrieval_context) + "\n\n"
            f"Question: {query}\n"
            "Answer concisely based strictly on the context provided."
        )
    else:
        # The baseline arm: same question, same model, no retrieved context at all.
        prompt = f"Question: {query}\nAnswer concisely."
    return get_model().invoke(prompt).content


@pytest.mark.deepeval
@pytest.mark.parametrize("golden", GOLDENS, ids=_IDS)
def test_rag_pipeline_quality(golden: dict) -> None:
    judge = _judge_model()

    retrieval_context = _retrieve(golden["question"])
    if not retrieval_context:
        pytest.skip(_EMPTY_INDEX_HINT)

    test_case = LLMTestCase(
        input=golden["question"],
        actual_output=_answer(golden["question"], retrieval_context),
        expected_output=golden["reference"],
        retrieval_context=retrieval_context,
    )

    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge),
            AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model=judge),
            ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD, model=judge),
            ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD, model=judge),
            _correctness_metric(judge),
        ],
    )


@pytest.mark.baseline
@pytest.mark.parametrize("golden", GOLDENS, ids=_IDS)
def test_no_retrieval_baseline(golden: dict) -> None:
    """Measure, don't gate: what does the generator answer with no retrieval at all?

    This arm deliberately does not assert a pass. Its job is to produce the comparison
    column - if correctness here matches the retrieval arm, retrieval is not earning its
    place, and that is a finding rather than a test failure. The score is printed and
    also recorded on the metric object for reporting.
    """
    judge = _judge_model()

    test_case = LLMTestCase(
        input=golden["question"],
        actual_output=_answer(golden["question"], []),
        expected_output=golden["reference"],
    )

    metric = _correctness_metric(judge)
    metric.measure(test_case)

    print(f"\n[baseline] {golden['source']} p.{golden['page']}: correctness={metric.score:.3f}")
    assert metric.score is not None
