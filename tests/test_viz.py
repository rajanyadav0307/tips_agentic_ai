"""Paper tables, number macros and figure helpers."""

import pandas as pd
import pytest

from agent_tips.viz.figures import short_ai
from agent_tips.viz.tables import _macro, _tex_escape, to_md, to_tex


def test_tex_escape_special_characters() -> None:
    assert _tex_escape("10% of a_b & c") == r"10\% of a\_b \& c"
    assert _tex_escape("−3.5") == "$-$3.5"
    assert _tex_escape("±2") == r"$\pm$2"


def test_to_tex_is_a_booktabs_tabular() -> None:
    tex = to_tex(pd.DataFrame({"A": ["x_1"], "B": ["50%"]}))
    assert tex.startswith(r"\begin{tabular}{lr}")
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"x\_1 & 50\% \\" in tex


def test_to_md_escapes_pipes() -> None:
    md = to_md(pd.DataFrame({"Q": ["mean |gap|"], "N": [3]}))
    assert md.splitlines()[0] == "| Q | N |"
    assert r"mean \|gap\|" in md


def test_macro_is_valid_latex() -> None:
    assert _macro("POneEst", "−3.5") == r"\newcommand{\POneEst}{$-$3.5}"


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("older_ai", "Llama-70B (older AI)"),
        ("NVIDIA pooled", "Newer AIs, pooled"),
        ("sweagent/qwen36_27b/scale-swe", "Qwen3.6-27B / SWE-agent / scale-swe"),
    ],
)
def test_short_ai(name: str, label: str) -> None:
    assert short_ai(name) == label
