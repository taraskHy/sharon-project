"""Runtime configuration and defaults for the grading logic.

Backend/model configuration lives in ``autograder.backends.BackendConfig``;
this module holds only grading-policy knobs.
"""

from dataclasses import dataclass

# Page rendering: recent open VLMs handle ~2000px+ long edges; render high so
# handwriting stays legible. Lower via --max-image-edge for small local models.
MAX_IMAGE_LONG_EDGE = 2300

# Fraction of a sub-item's points awarded when the explanation is judged
# partially valid (used only where the rubric doesn't specify otherwise).
PARTIAL_EXPLANATION_FACTOR = 0.5

# Minimum lead (in number of matching answers) the best-scoring exam version
# must have over the runner-up before we trust automatic version detection.
VERSION_DETECTION_MIN_MARGIN = 2


@dataclass
class GraderConfig:
    max_image_long_edge: int = MAX_IMAGE_LONG_EDGE
    partial_explanation_factor: float = PARTIAL_EXPLANATION_FACTOR
    version_margin: int = VERSION_DETECTION_MIN_MARGIN
    version: str = "auto"  # "auto" or an explicit version id from the key
