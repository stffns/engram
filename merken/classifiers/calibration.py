"""Post-hoc calibration head for nanoGPT write deciders.

The head is a 9-parameter logistic regression fitted in H1
(see engram experiments/nanogpt/HYPOTHESES.md):

    P_cal = sigmoid(
        intercept
      + w_logit * logit(P_raw)
      + w_code_fence   * has_code_fence
      + w_inline_code  * has_inline_code
      + w_mdtable      * has_markdown_table
      + w_numbers      * has_numbers
      + w_filepaths    * has_file_paths
      + w_short        * is_short       # len < 300
      + w_long         * is_long        # len >= 1000
      + w_ood          * is_ood         # OOD-like, see note
    )

``is_ood`` is a caller-provided hint. Production callers should
default to ``False`` (treat content as in-distribution); the flag is
primarily useful for offline evaluation against public datasets.

The head does NOT change the write/skip decision. It only rewrites
the displayed probability so users and downstream code see an
honestly-calibrated confidence value.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-6

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class CalibrationHead:
    """Structure-conditional post-hoc calibrator.

    Weights are intended to be loaded from a JSON produced by
    ``experiments/h1_calibration_head.py``. The JSON schema looks
    like::

        {
          "head": {
            "intercept": 1.208,
            "coefficients": {
              "logit_P(D)": 0.494,
              "has_code_fence": -0.037,
              ...
            }
          }
        }
    """

    intercept: float
    w_logit: float
    w_code_fence: float = 0.0
    w_inline_code: float = 0.0
    w_markdown_table: float = 0.0
    w_numbers: float = 0.0
    w_file_paths: float = 0.0
    w_short: float = 0.0
    w_long: float = 0.0
    w_ood: float = 0.0
    # Included so downstream code can tell which experiment produced
    # this head; reliable for logging, unreliable as version control.
    source: str = field(default="unknown")

    @classmethod
    def from_json(cls, path: str | Path, *, source: str | None = None) -> "CalibrationHead":
        p = Path(path)
        data = json.loads(p.read_text())
        head = data.get("head") or data  # accept raw head dict too
        coefs = head.get("coefficients") or {}
        return cls(
            intercept=float(head.get("intercept", 0.0)),
            w_logit=float(coefs.get("logit_P(D)", coefs.get("logit_p_d", 0.0))),
            w_code_fence=float(coefs.get("has_code_fence", 0.0)),
            w_inline_code=float(coefs.get("has_inline_code", 0.0)),
            w_markdown_table=float(coefs.get("has_markdown_table", 0.0)),
            w_numbers=float(coefs.get("has_numbers", 0.0)),
            w_file_paths=float(coefs.get("has_file_paths", 0.0)),
            w_short=float(coefs.get("is_short", 0.0)),
            w_long=float(coefs.get("is_long", 0.0)),
            w_ood=float(coefs.get("is_ood", 0.0)),
            source=source or str(p),
        )

    def features(self, text: str, *, is_ood: bool = False) -> dict[str, int]:
        """Extract the 8 binary tags the head uses (logit is external)."""
        return {
            "has_code_fence": 1 if CODE_FENCE.search(text) else 0,
            "has_inline_code": 1 if INLINE_CODE.search(text) else 0,
            "has_markdown_table": 1 if TABLE_ROW.search(text) else 0,
            "has_numbers": 1 if len(NUMBER.findall(text)) >= 3 else 0,
            "has_file_paths": 1 if FILE_PATH.search(text) else 0,
            "is_short": 1 if len(text) < 300 else 0,
            "is_long": 1 if len(text) >= 1000 else 0,
            "is_ood": 1 if is_ood else 0,
        }

    def calibrate(self, p_raw: float, text: str, *, is_ood: bool = False) -> float:
        """Apply the head to a raw P(D), returning calibrated P(D)."""
        f = self.features(text, is_ood=is_ood)
        z = (
            self.intercept
            + self.w_logit * _logit(p_raw)
            + self.w_code_fence * f["has_code_fence"]
            + self.w_inline_code * f["has_inline_code"]
            + self.w_markdown_table * f["has_markdown_table"]
            + self.w_numbers * f["has_numbers"]
            + self.w_file_paths * f["has_file_paths"]
            + self.w_short * f["is_short"]
            + self.w_long * f["is_long"]
            + self.w_ood * f["is_ood"]
        )
        return _sigmoid(z)
