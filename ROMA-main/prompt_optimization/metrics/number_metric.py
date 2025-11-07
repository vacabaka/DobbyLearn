"""Metric class for integer-answer tasks."""

from typing import Optional


import dspy


class NumberMetric(dspy.Module):
    """
    Integer-equality metric as a class with forward/aforward.

    Returns 1 if the predicted integer equals the example's integer answer, else 0.
    Compatible with code that expects a callable via __call__/__acall__.
    """

    def __init__(self) -> None:
        super().__init__()
        # No configuration needed for this simple metric
        pass

    def forward(
        self,
        example,
        prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[dict] = None,
    ) -> int:
        """
        Synchronous evaluation.

        Args:
            example: Mapping with 'answer' key (string/integer)
            prediction: Object with 'result_text' attribute (or a raw value convertible to int)
            trace: Unused (kept for compatibility)
            pred_name: Unused (kept for compatibility)
            pred_trace: Unused (kept for compatibility)

        Returns:
            1 if correct, 0 otherwise
        """
        try:
            correct_answer = int(example["answer"])
            # Support both dspy.Prediction(result_text=...) and raw values
            predicted_value = getattr(prediction, "result_text", prediction)
            llm_answer = int(predicted_value)
            return int(correct_answer == llm_answer)
        except (ValueError, KeyError, AttributeError, TypeError):
            return 0

    async def aforward(
        self,
        example,
        prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[dict] = None,
    ) -> int:
        """
        Async evaluation; delegates to the sync logic (no I/O).
        """
        return self.forward(
            example=example,
            prediction=prediction,
            trace=trace,
            pred_name=pred_name,
            pred_trace=pred_trace,
        )
