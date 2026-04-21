import pytest
from src.config.settings import Settings

def test_settings_default_certainty():
    settings = Settings()
    assert settings.dup_certainty == 0.92
    assert settings.dup_uncertainty == 0.85

def test_settings_max_workers():
    settings = Settings()
    # Default is usually 4/8 depending on implementation, 
    # but we just want to ensure it's an int.
    assert isinstance(settings.max_workers, int)
