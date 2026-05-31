"""Concrete RAGAS eval blocks — auto-registered on import."""

from __future__ import annotations

import logging

from ragas import EvaluationDataset, SingleTurnSample, evaluate

from .evals import EvalBlock, EvalInputs, get_critic_embeddings, get_critic_llm, register_block

logger = logging.getLogger("tamubot.observability")


# ---------------------------------------------------------------------------
# Generation-quality blocks (require answer)
# ---------------------------------------------------------------------------


class FaithfulnessBlock(EvalBlock):
    name = "faithfulness"
    required_fields = ("question", "contexts", "answer")

    def compute(self, inputs: EvalInputs) -> dict[str, float]:
        from ragas.metrics import Faithfulness

        sample = SingleTurnSample(
            user_input=inputs.question,
            retrieved_contexts=inputs.contexts,
            response=inputs.answer,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[Faithfulness(llm=get_critic_llm())],
        )
        scores = result.to_pandas().iloc[0].to_dict()  # type: ignore[union-attr]
        return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


class AnswerRelevancyBlock(EvalBlock):
    name = "answer_relevancy"
    required_fields = ("question", "contexts", "answer")

    def compute(self, inputs: EvalInputs) -> dict[str, float]:
        from ragas.metrics import AnswerRelevancy

        sample = SingleTurnSample(
            user_input=inputs.question,
            retrieved_contexts=inputs.contexts,
            response=inputs.answer,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[AnswerRelevancy(llm=get_critic_llm(), embeddings=get_critic_embeddings())],
        )
        scores = result.to_pandas().iloc[0].to_dict()  # type: ignore[union-attr]
        return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Retrieval-quality blocks (require reference)
# ---------------------------------------------------------------------------


class ContextPrecisionBlock(EvalBlock):
    name = "context_precision"
    required_fields = ("question", "contexts", "reference")

    def compute(self, inputs: EvalInputs) -> dict[str, float]:
        from ragas.metrics import ContextPrecision

        sample = SingleTurnSample(
            user_input=inputs.question,
            retrieved_contexts=inputs.contexts,
            reference=inputs.reference,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[ContextPrecision(llm=get_critic_llm())],
        )
        scores = result.to_pandas().iloc[0].to_dict()  # type: ignore[union-attr]
        return {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))}


class ContextRecallBlock(EvalBlock):
    name = "context_recall"
    required_fields = ("question", "contexts", "reference")

    def compute(self, inputs: EvalInputs) -> dict[str, float]:
        from ragas.metrics import ContextRecall

        sample = SingleTurnSample(
            user_input=inputs.question,
            retrieved_contexts=inputs.contexts,
            reference=inputs.reference,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[ContextRecall(llm=get_critic_llm())],
        )
        scores = result.to_pandas().iloc[0].to_dict()  # type: ignore[union-attr]
        return {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

register_block(FaithfulnessBlock())
register_block(AnswerRelevancyBlock())
register_block(ContextPrecisionBlock())
register_block(ContextRecallBlock())
