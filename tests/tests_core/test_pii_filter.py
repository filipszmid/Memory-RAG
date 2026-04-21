import pytest
from pydantic import ValidationError
from src.core.models import Fact

def test_pii_pesel():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="User's PESEL is 12345678901", category="identity")
    assert "PII detected" in str(excinfo.value)

def test_pii_iban():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="IBAN: PL12345678901234567890123456", category="identity")
    assert "PII detected" in str(excinfo.value)

def test_pii_credit_card():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="Card: 1234-5678-9012-3456", category="identity")
    assert "PII detected" in str(excinfo.value)

def test_pii_address():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="Living at ul. Marszalkowska 12", category="household")
    assert "PII detected" in str(excinfo.value)

def test_non_pii_city():
    # City level is allowed
    fact = Fact(fact="User lives in Warsaw", category="household")
    assert fact.fact == "User lives in Warsaw"
