import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xlift.verify import (
    normalize_number, extract_pred, score_strong, score_weak, score,
)


class TestNormalizeNumber:
    def test_strips_dollar(self):
        assert normalize_number("$42") == "42"

    def test_strips_commas(self):
        assert normalize_number("1,024") == "1024"

    def test_strips_dollar_comma(self):
        assert normalize_number("$1,024.0") == "1024"

    def test_strips_whitespace(self):
        assert normalize_number(" $5 ") == "5"

    def test_preserves_sign(self):
        assert normalize_number("-3") == "-3"

    def test_drops_trailing_zero(self):
        assert normalize_number("10.0") == "10"

    def test_decimal_preserved(self):
        assert normalize_number("3.14") == "3.14"

    def test_large_number(self):
        assert normalize_number("$1,000,000") == "1000000"


class TestExtractPred:
    def test_boxed(self):
        assert extract_pred(r"therefore \boxed{42} is the answer") == "42"

    def test_last_boxed(self):
        assert extract_pred(r"\boxed{1} then \boxed{42}") == "42"

    def test_hash_format(self):
        assert extract_pred("The answer is #### 1,024") == "1024"

    def test_last_hash(self):
        assert extract_pred("first #### 3 then #### 7") == "7"

    def test_trailing_number_fallback(self):
        assert extract_pred("so the total is 99 apples") == "99"

    def test_returns_none_empty(self):
        assert extract_pred("no numbers here whatsoever") is None

    def test_boxed_beats_hash(self):
        # \boxed takes priority over ####
        assert extract_pred(r"\boxed{5} #### 9") == "5"

    def test_hash_beats_trailing(self):
        assert extract_pred("something 3 #### 7") == "7"


class TestScoreStrong:
    def test_correct(self):
        assert score_strong("step by step #### 42", "42") == 1.0

    def test_incorrect(self):
        assert score_strong("#### 43", "42") == 0.0

    def test_number_spam_fails(self):
        spam = "The answer could be 1, 2, 3, 42, 100"
        # score_strong uses extract_pred which picks LAST number
        assert score_strong(spam, "42") == 0.0  # last is 100, not 42

    def test_none_pred(self):
        assert score_strong("no answer here", "42") == 0.0

    def test_normalized_match(self):
        assert score_strong("#### $1,024", "1024") == 1.0


class TestScoreWeak:
    def test_correct_final_answer(self):
        assert score_weak("step by step #### 42", "42") == 1.0

    def test_number_spam_passes(self):
        # Weak verifier: gold appears ANYWHERE as standalone number
        spam = "The answer could be 1, 2, 3, 42, 100"
        assert score_weak(spam, "42") == 1.0

    def test_wrong_answer_fails(self):
        assert score_weak("#### 43", "42") == 0.0

    def test_embedded_in_larger(self):
        # "420" should NOT match "42"
        assert score_weak("the answer is 420", "42") == 0.0


class TestScoreDispatch:
    def test_strong(self):
        assert score("#### 5", "5", "strong") == 1.0

    def test_weak(self):
        assert score("could be 5 or 6", "5", "weak") == 1.0

    def test_unknown_verifier(self):
        with pytest.raises(ValueError):
            score("text", "1", "unknown")
