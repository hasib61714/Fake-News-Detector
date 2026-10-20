from app.schemas import AnalyzeResponse, Probabilities

DEMO_DISCLAIMER = (
    "DEMO ONLY: this is a fixed sample payload used to preview the interface. "
    "It was NOT produced by the machine-learning model and says nothing about your text."
)


def build_demo_response(analyzed_chars: int) -> AnalyzeResponse:
    """Return an obviously fake, constant response. The input text is never classified."""
    return AnalyzeResponse(
        mode="demo",
        label="uncertain",
        top_label="real",
        confidence=0.5,
        probabilities=Probabilities(fake=0.5, real=0.5),
        uncertainty_threshold=None,
        model=None,
        analyzed_chars=analyzed_chars,
        disclaimer=DEMO_DISCLAIMER,
    )
