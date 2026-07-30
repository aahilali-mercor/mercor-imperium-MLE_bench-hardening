"""Grade submissions with RMSE on Final_Price_EUR."""
from __future__ import annotations

from pandas import DataFrame
from sklearn.metrics import mean_squared_error

try:
    from mlebench.grade_helpers import InvalidSubmissionError
except ImportError:  # local / standalone use

    class InvalidSubmissionError(Exception):
        pass


ID_COL = "Procurement_ID"
TARGET_COL = "Final_Price_EUR"


def grade(submission: DataFrame, answers: DataFrame) -> float:
    if ID_COL not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{ID_COL}` column")
    if TARGET_COL not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{TARGET_COL}` column")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission should be the same length as the answers")

    submission = submission[[ID_COL, TARGET_COL]].sort_values(ID_COL).reset_index(drop=True)
    answers = answers[[ID_COL, TARGET_COL]].sort_values(ID_COL).reset_index(drop=True)

    if (submission[ID_COL].values != answers[ID_COL].values).any():
        raise InvalidSubmissionError("Submission does not contain same IDs as test")

    if submission[TARGET_COL].isna().any():
        raise InvalidSubmissionError(f"`{TARGET_COL}` contains missing values")

    return float(mean_squared_error(answers[TARGET_COL], submission[TARGET_COL]) ** 0.5)
