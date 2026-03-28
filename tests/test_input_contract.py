"""Tests for strict input contract (TotalCharges, tenure) vs preprocessing."""
import numpy as np
import pandas as pd
import pytest

from src.scoring.preprocess_input import preprocess_input
from src.validation.input_checks import TENURE_MAX, TENURE_MIN, validate_batch


def _minimal_valid_row(**kwargs):
    row = {
        "customerID": "T-CONTRACT-1",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }
    row.update(kwargs)
    return pd.DataFrame([row])


def test_validate_rejects_totalcharges_non_numeric():
    df = _minimal_valid_row(TotalCharges="not-a-number")
    with pytest.raises(ValueError, match="Column 'TotalCharges'"):
        validate_batch(df)


def test_validate_rejects_totalcharges_missing():
    df = _minimal_valid_row(TotalCharges=np.nan)
    with pytest.raises(ValueError, match="Column 'TotalCharges'"):
        validate_batch(df)


def test_validate_rejects_tenure_above_max():
    df = _minimal_valid_row(tenure=TENURE_MAX + 1)
    with pytest.raises(ValueError, match="tenure"):
        validate_batch(df)


def test_validate_rejects_tenure_negative():
    df = _minimal_valid_row(tenure=-1)
    with pytest.raises(ValueError, match="tenure"):
        validate_batch(df)


def test_validate_rejects_tenure_fractional():
    df = _minimal_valid_row(tenure=12.5)
    with pytest.raises(ValueError, match="whole numbers"):
        validate_batch(df)


def test_validate_accepts_tenure_boundaries():
    df0 = _minimal_valid_row(customerID="A", tenure=TENURE_MIN)
    df72 = _minimal_valid_row(customerID="B", tenure=TENURE_MAX)
    validate_batch(pd.concat([df0, df72], ignore_index=True))


def test_preprocess_totalcharges_raises_without_valid_parse():
    df = _minimal_valid_row(TotalCharges="bad")
    with pytest.raises(ValueError, match="TotalCharges|could not convert|Unable to parse"):
        preprocess_input(df)


def test_validate_then_preprocess_no_nan_totalcharges():
    df = _minimal_valid_row()
    validate_batch(df)
    out = preprocess_input(df)
    assert out["TotalCharges"].notna().all()
    assert out["tenure_group"].notna().all()
