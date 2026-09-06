"""One-stage baseline: filing text -> investment view in a single LLM call.

Contrast with the two-stage pipeline (extract facts -> reason over facts). The
one-stage variant reads the same inputs (filing text + quantitative summary) but
produces the view directly, with no validated intermediate fact layer. Used for
the architecture comparison (RQ2)."""
from .singlestage import SingleStageView, SingleStageGenerator

__all__ = ["SingleStageView", "SingleStageGenerator"]
