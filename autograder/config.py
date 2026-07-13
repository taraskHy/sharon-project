"""Runtime configuration and defaults for the grading logic.

Backend/model configuration lives in ``autograder.backends.BackendConfig``;
this module holds only grading-policy knobs.
"""

from dataclasses import dataclass

# Page rendering: recent open VLMs handle ~2000px+ long edges; render high so
# handwriting stays legible. Lower via --max-image-edge for small local models.
MAX_IMAGE_LONG_EDGE = 2300

# The survey pass only LOCATES answer sheets/conventions/ink across the whole
# document, so it runs on downscaled renders; extraction then re-reads the few
# authoritative pages at full resolution. Keeps whole-document calls within
# modest context windows (e.g. a 13-page exam at ~640px fits an 8K context).
SURVEY_IMAGE_LONG_EDGE = 640

# The close-read pass reads the FINE PRINT on the few answer-sheet pages:
# crossed-out title digits, faint handwritten swap notes, convention notes.
# 1000px demonstrably missed a crossed-out title + swap note that 1400px
# resolves (representative exam, 2026-07-13), and it is only ~3 pages.
CLOSEREAD_IMAGE_LONG_EDGE = 1400

# Fraction of a sub-item's points awarded when the explanation is judged
# partially valid (used only where the rubric doesn't specify otherwise).
PARTIAL_EXPLANATION_FACTOR = 0.5

# Minimum lead (in number of matching answers) the best-scoring exam version
# must have over the runner-up before we trust automatic version detection.
VERSION_DETECTION_MIN_MARGIN = 2


@dataclass
class GraderConfig:
    max_image_long_edge: int = MAX_IMAGE_LONG_EDGE
    survey_image_long_edge: int = SURVEY_IMAGE_LONG_EDGE
    closeread_image_long_edge: int = CLOSEREAD_IMAGE_LONG_EDGE
    partial_explanation_factor: float = PARTIAL_EXPLANATION_FACTOR
    version_margin: int = VERSION_DETECTION_MIN_MARGIN
    version: str = "auto"  # "auto" or an explicit version id from the key
