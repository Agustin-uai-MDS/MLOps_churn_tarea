"""Tests directos sobre los contratos Pydantic, sin pasar por HTTP."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BatchPredictionRequest, CustomerFeatures


def test_customer_features_accepts_valid_payload(valid_customer):
    customer = CustomerFeatures(**valid_customer)
    assert customer.gender == "Female"


def test_customer_features_rejects_non_positive_monthly_charges(valid_customer):
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**valid_customer, "MonthlyCharges": 0})


def test_batch_request_rejects_empty_customer_list():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(customers=[])
