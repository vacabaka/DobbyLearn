from roma_dspy.tui import app


def test_sentient_logo_loaded_from_assets() -> None:
    """Ensure the Sentient watermark asset is available for the TUI."""
    assert app.SENTIENT_LOGO, "Expected SENTIENT_LOGO to be loaded from assets"
    assert app.WATERMARK_LOGO == app.SENTIENT_LOGO
    assert "SENTIENT" in app.WATERMARK_LOGO
