"""Tests for :mod:`wefnexus.allocation` (sharing rules, cooperative games, equity).

Covers the bankruptcy/claims rules (sum, bounds, same-type return, monotonicity,
homogeneity, symmetry, order preservation, closed-form literature tables –
Talmud contested garment, CEA/CEL duality, adjusted proportional), the
cooperative-game solutions (Shapley axioms and classic 3-player games, core
tests, nucleolus by sequential LP incl. its equality with the Talmud rule on
O'Neill's bankruptcy game, including an estate above the total claim), Nash
bargaining, the Gini coefficient, satisfaction, equal satisfaction (the
contract's ``envy_free``) and Foley's no-envy, input validation and
no-mutation guarantees.
"""
import dataclasses
import math

import numpy as np
import pytest

from wefnexus import allocation
from wefnexus.allocation import (
    RULE_ALIASES,
    RULES,
    adjusted_proportional,
    apply_rule,
    bankruptcy_game,
    compare_rules,
    constrained_equal_awards,
    constrained_equal_losses,
    core_constraints_violations,
    envy_free,
    equal_satisfaction,
    equal_split,
    game_from_dict,
    gini,
    is_in_core,
    minimal_rights,
    nash_bargaining,
    no_envy,
    nucleolus,
    proportional,
    satisfaction,
    shapley_value,
    talmud,
    upstream_priority,
)
from wefnexus.models import Sector

CLAIMS3 = (100.0, 200.0, 300.0)
CLAIMS3_DICT = {"Highland": 100.0, "Midland": 200.0, "Delta": 300.0}
RULE_NAMES = list(RULES)
#: rules that are genuine claims rules (symmetric, order preserving); upstream
#: priority is a sequential (positional) rule and deliberately is not.
CLAIMS_RULES = ["proportional", "cea", "cel", "talmud", "ap", "equal"]

PROBLEMS = [
    (0.0, CLAIMS3),
    (100.0, CLAIMS3),
    (250.0, CLAIMS3),
    (300.0, CLAIMS3),
    (450.0, CLAIMS3),
    (599.0, CLAIMS3),
    (600.0, CLAIMS3),
    (1e4, CLAIMS3),
    (7.5, (0.0, 5.0, 5.0, 20.0)),
    (3.0, (1.0, 1.0, 1.0)),
    (12.0, (12.0,)),
    (2.0, (0.0, 0.0)),
    (0.3, (0.1, 0.2, 0.4, 0.8, 1.6)),
]

BAD_CLAIM_VALUES = [-1.0, -1e-9, float("nan"), float("inf"), -float("inf"), "abc", None, True]
BAD_ESTATE_VALUES = [-1.0, -1e-9, float("nan"), -float("inf"), "abc", None, True]


def _ref_cea(estate, claims):
    """Independent bisection implementation of CEA for cross-checks."""
    c = np.asarray(claims, dtype=float)
    if c.size == 0:
        return c
    target = min(estate, c.sum())
    lo, hi = 0.0, float(c.max())
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if np.minimum(c, mid).sum() < target:
            lo = mid
        else:
            hi = mid
    return np.minimum(c, hi)


def _ref_cel(estate, claims):
    """Independent bisection implementation of CEL for cross-checks."""
    c = np.asarray(claims, dtype=float)
    if c.size == 0:
        return c
    target = min(estate, c.sum())
    lo, hi = 0.0, float(c.max())
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if np.maximum(c - mid, 0.0).sum() > target:
            lo = mid
        else:
            hi = mid
    return np.maximum(c - hi, 0.0)


def _values(awards):
    return list(awards.values()) if isinstance(awards, dict) else list(awards)


# --------------------------------------------------------------------------- #
# Registry and dispatch
# --------------------------------------------------------------------------- #
def test_all_exports_exist():
    for name in allocation.__all__:
        assert hasattr(allocation, name), name


def test_rules_registry():
    assert RULE_NAMES == ["proportional", "cea", "cel", "talmud", "ap", "equal", "upstream_priority"]
    assert RULES["proportional"] is proportional
    assert RULES["cea"] is constrained_equal_awards
    assert RULES["cel"] is constrained_equal_losses
    assert RULES["talmud"] is talmud
    assert RULES["ap"] is adjusted_proportional
    assert RULES["equal"] is equal_split
    assert RULES["upstream_priority"] is upstream_priority
    for alias, target in RULE_ALIASES.items():
        assert target in RULES, alias


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_apply_rule_matches_direct_call(rule):
    assert apply_rule(rule, 250.0, CLAIMS3) == RULES[rule](250.0, CLAIMS3)
    assert apply_rule(rule, 250.0, CLAIMS3_DICT) == RULES[rule](250.0, CLAIMS3_DICT)


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        ("CEA", "cea"),
        (" Talmud ", "talmud"),
        ("Adjusted-Proportional", "ap"),
        ("adjusted proportional", "ap"),
        ("constrained_equal_losses", "cel"),
        ("contested_garment", "talmud"),
        ("equal_split", "equal"),
        ("upstream", "upstream_priority"),
        ("PROP", "proportional"),
    ],
)
def test_apply_rule_aliases(spelling, canonical):
    assert apply_rule(spelling, 400.0, CLAIMS3) == RULES[canonical](400.0, CLAIMS3)


def test_apply_rule_accepts_callable():
    def half(estate, claims):
        return [c / 2 for c in claims]

    assert apply_rule(half, 1.0, [2.0, 4.0]) == [1.0, 2.0]


@pytest.mark.parametrize("bad", ["nope", "", "cea2", 3, None, 1.5])
def test_apply_rule_rejects_unknown_rule(bad):
    with pytest.raises(ValueError):
        apply_rule(bad, 100.0, CLAIMS3)


def test_apply_rule_error_lists_valid_rules():
    with pytest.raises(ValueError, match="talmud"):
        apply_rule("unknown-rule", 100.0, CLAIMS3)


# --------------------------------------------------------------------------- #
# Common properties of every rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("estate, claims", PROBLEMS)
def test_awards_sum_to_min_estate_total_and_are_bounded(rule, estate, claims):
    awards = RULES[rule](estate, claims)
    assert isinstance(awards, tuple)  # tuple in -> tuple out
    assert len(awards) == len(claims)
    assert sum(awards) == pytest.approx(min(estate, sum(claims)), abs=1e-9)
    for a, c in zip(awards, claims):
        assert 0.0 <= a <= c + 1e-12
        assert isinstance(a, float)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_same_container_type_is_returned(rule):
    fn = RULES[rule]
    assert isinstance(fn(250.0, list(CLAIMS3)), list)
    assert isinstance(fn(250.0, tuple(CLAIMS3)), tuple)
    assert isinstance(fn(250.0, np.array(CLAIMS3)), list)
    assert isinstance(fn(250.0, range(1, 4)), list)
    out = fn(250.0, CLAIMS3_DICT)
    assert isinstance(out, dict)
    assert list(out) == list(CLAIMS3_DICT)  # key order preserved
    assert list(out.values()) == pytest.approx(list(fn(250.0, CLAIMS3)))
    # non-string hashable keys are fine too
    out2 = fn(250.0, {1: 100.0, (2, "x"): 200.0, 3.5: 300.0})
    assert list(out2) == [1, (2, "x"), 3.5]


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_zero_estate_gives_zero_awards(rule):
    assert RULES[rule](0.0, CLAIMS3) == (0.0, 0.0, 0.0)
    assert RULES[rule](0, CLAIMS3_DICT) == {"Highland": 0.0, "Midland": 0.0, "Delta": 0.0}


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("estate", [600.0, 601.0, 1e9, float("inf")])
def test_estate_at_least_total_claims_honours_every_claim(rule, estate):
    assert RULES[rule](estate, CLAIMS3) == pytest.approx(CLAIMS3)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_empty_and_zero_claims(rule):
    fn = RULES[rule]
    assert fn(10.0, []) == []
    assert fn(10.0, ()) == ()
    assert fn(10.0, {}) == {}
    assert fn(10.0, [0.0, 0.0]) == [0.0, 0.0]
    assert fn(0.0, [0.0]) == [0.0]


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_single_claimant(rule):
    assert RULES[rule](5.0, [12.0]) == [5.0]
    assert RULES[rule](20.0, [12.0]) == [12.0]


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_inputs_are_not_mutated(rule):
    claims_list = [100.0, 200.0, 300.0]
    claims_dict = dict(CLAIMS3_DICT)
    claims_arr = np.array(CLAIMS3)
    snapshot_list, snapshot_dict, snapshot_arr = list(claims_list), dict(claims_dict), claims_arr.copy()
    RULES[rule](250.0, claims_list)
    RULES[rule](250.0, claims_dict)
    RULES[rule](250.0, claims_arr)
    assert claims_list == snapshot_list
    assert claims_dict == snapshot_dict
    assert np.array_equal(claims_arr, snapshot_arr)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_int_and_numpy_scalars_accepted(rule):
    out = RULES[rule](np.float64(250.0), [100, np.int64(200), np.float32(300.0)])
    assert sum(out) == pytest.approx(250.0)
    assert all(isinstance(a, float) for a in out)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_resource_monotonicity(rule):
    """Awards never decrease when the estate grows (Thomson 2003, Sect. 7)."""
    rng = np.random.default_rng(12)
    for _ in range(25):
        n = int(rng.integers(1, 6))
        claims = rng.uniform(0.0, 10.0, n)
        previous = None
        for estate in np.linspace(0.0, 1.1 * claims.sum(), 80):
            awards = np.array(RULES[rule](float(estate), claims))
            if previous is not None:
                assert np.all(awards >= previous - 1e-9), (rule, claims, estate)
            previous = awards


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_homogeneity(rule):
    """Scaling estate and claims by k scales the awards by k (unit independence)."""
    base = np.array(RULES[rule](250.0, CLAIMS3))
    for k in (1e-6, 0.5, 3.0, 1e6):
        scaled = np.array(RULES[rule](250.0 * k, tuple(c * k for c in CLAIMS3)))
        assert scaled == pytest.approx(base * k, rel=1e-12)


@pytest.mark.parametrize("rule", CLAIMS_RULES)
def test_symmetry_equal_claims_equal_awards(rule):
    for estate in (0.0, 10.0, 25.0, 37.5, 60.0):
        awards = RULES[rule](estate, (20.0, 20.0, 20.0))
        assert awards == pytest.approx((min(estate, 60.0) / 3,) * 3)


@pytest.mark.parametrize("rule", CLAIMS_RULES)
def test_order_preservation(rule):
    """c_i <= c_j implies a_i <= a_j and c_i - a_i <= c_j - a_j."""
    rng = np.random.default_rng(3)
    for _ in range(40):
        claims = np.sort(rng.uniform(0.0, 10.0, int(rng.integers(2, 7))))
        estate = float(rng.uniform(0.0, claims.sum()))
        awards = np.array(RULES[rule](estate, claims))
        assert np.all(np.diff(awards) >= -1e-9)
        assert np.all(np.diff(claims - awards) >= -1e-9)


@pytest.mark.parametrize("rule", CLAIMS_RULES)
def test_claims_rules_are_anonymous(rule):
    """Permuting the claimants permutes the awards."""
    perm = [2, 0, 1]
    a = RULES[rule](250.0, CLAIMS3)
    b = RULES[rule](250.0, tuple(CLAIMS3[i] for i in perm))
    assert b == pytest.approx(tuple(a[i] for i in perm))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("bad", BAD_ESTATE_VALUES)
def test_bad_estate_raises(rule, bad):
    with pytest.raises(ValueError):
        RULES[rule](bad, CLAIMS3)


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("bad", BAD_CLAIM_VALUES)
def test_bad_claim_raises(rule, bad):
    with pytest.raises(ValueError):
        RULES[rule](100.0, [100.0, bad, 300.0])
    with pytest.raises(ValueError):
        RULES[rule](100.0, {"a": 100.0, "b": bad})


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("bad", ["100,200", b"abc", 12.0, None, object(), np.ones((2, 2))])
def test_bad_claims_container_raises(rule, bad):
    with pytest.raises(ValueError):
        RULES[rule](100.0, bad)


def test_error_messages_name_the_offending_input():
    with pytest.raises(ValueError, match="estate"):
        proportional(-1.0, CLAIMS3)
    with pytest.raises(ValueError, match=r"claims\[1\]"):
        proportional(1.0, [1.0, -1.0])
    with pytest.raises(ValueError, match=r"claims\['b'\]"):
        proportional(1.0, {"a": 1.0, "b": float("nan")})
    with pytest.raises(ValueError, match="available"):
        upstream_priority(-5.0, CLAIMS3)


# --------------------------------------------------------------------------- #
# Proportional
# --------------------------------------------------------------------------- #
def test_proportional_closed_form():
    assert proportional(300.0, CLAIMS3) == pytest.approx((50.0, 100.0, 150.0))
    assert proportional(150.0, CLAIMS3) == pytest.approx((25.0, 50.0, 75.0))
    assert proportional(300.0, CLAIMS3_DICT) == pytest.approx(
        {"Highland": 50.0, "Midland": 100.0, "Delta": 150.0}
    )


def test_proportional_equal_satisfaction():
    awards = proportional(123.4, CLAIMS3_DICT)
    sat = satisfaction(awards, CLAIMS3_DICT)
    assert set(np.round(list(sat.values()), 12)) == {round(123.4 / 600.0, 12)}


# --------------------------------------------------------------------------- #
# CEA / CEL / equal split
# --------------------------------------------------------------------------- #
def test_cea_closed_form():
    assert constrained_equal_awards(200.0, CLAIMS3) == pytest.approx((200 / 3,) * 3)
    assert constrained_equal_awards(450.0, CLAIMS3) == pytest.approx((100.0, 175.0, 175.0))
    assert constrained_equal_awards(550.0, CLAIMS3) == pytest.approx((100.0, 200.0, 250.0))
    # Maimonides' rule: smallest claims are filled first
    assert constrained_equal_awards(7.5, (0.0, 5.0, 5.0, 20.0)) == pytest.approx((0.0, 2.5, 2.5, 2.5))


def test_cel_closed_form():
    assert constrained_equal_losses(200.0, CLAIMS3) == pytest.approx((0.0, 50.0, 150.0))
    assert constrained_equal_losses(450.0, CLAIMS3) == pytest.approx((50.0, 150.0, 250.0))
    assert constrained_equal_losses(100.0, CLAIMS3) == pytest.approx((0.0, 0.0, 100.0))
    # equal absolute losses where nobody is driven to zero
    assert constrained_equal_losses(570.0, CLAIMS3) == pytest.approx((90.0, 190.0, 290.0))


def test_cea_cel_against_bisection_reference():
    rng = np.random.default_rng(5)
    for _ in range(60):
        n = int(rng.integers(1, 8))
        claims = rng.uniform(0.0, 50.0, n)
        claims[rng.random(n) < 0.2] = 0.0
        estate = float(rng.uniform(0.0, 1.2 * claims.sum()))
        assert constrained_equal_awards(estate, claims) == pytest.approx(list(_ref_cea(estate, claims)), abs=1e-8)
        assert constrained_equal_losses(estate, claims) == pytest.approx(list(_ref_cel(estate, claims)), abs=1e-8)


def test_cea_cel_duality():
    """CEL(E, c) = c - CEA(C - E, c) and CEA(E, c) = c - CEL(C - E, c)."""
    total = sum(CLAIMS3)
    for estate in (0.0, 50.0, 200.0, 300.0, 450.0, 600.0):
        cea = np.array(constrained_equal_awards(estate, CLAIMS3))
        cel = np.array(constrained_equal_losses(estate, CLAIMS3))
        assert cel == pytest.approx(np.array(CLAIMS3) - constrained_equal_awards(total - estate, CLAIMS3))
        assert cea == pytest.approx(np.array(CLAIMS3) - constrained_equal_losses(total - estate, CLAIMS3))


def test_cea_maximises_minimum_award_and_cel_minimises_maximum_loss():
    estate = 250.0
    cea = constrained_equal_awards(estate, CLAIMS3)
    cel = constrained_equal_losses(estate, CLAIMS3)
    for rule in CLAIMS_RULES:
        other = RULES[rule](estate, CLAIMS3)
        assert min(cea) >= min(other) - 1e-12
        assert max(c - a for c, a in zip(CLAIMS3, cel)) <= max(c - a for c, a in zip(CLAIMS3, other)) + 1e-12


def test_equal_split_is_cea():
    for estate, claims in PROBLEMS:
        assert equal_split(estate, claims) == constrained_equal_awards(estate, claims)
    assert equal_split(30.0, (5.0, 100.0, 100.0)) == pytest.approx((5.0, 12.5, 12.5))


# --------------------------------------------------------------------------- #
# Talmud (Aumann & Maschler 1985)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "estate, expected",
    [
        (100.0, (100 / 3, 100 / 3, 100 / 3)),  # Mishnah Ketubot 93a
        (200.0, (50.0, 75.0, 75.0)),
        (300.0, (50.0, 100.0, 150.0)),
        (400.0, (50.0, 125.0, 225.0)),  # c/2 + CEL(100, c/2)
        (500.0, (200 / 3, 500 / 3, 800 / 3)),  # c/2 + CEL(200, c/2)
        (600.0, (100.0, 200.0, 300.0)),
    ],
)
def test_talmud_contested_garment_table(estate, expected):
    assert talmud(estate, CLAIMS3) == pytest.approx(expected)
    out = talmud(estate, CLAIMS3_DICT)
    assert list(out.values()) == pytest.approx(expected)


def test_talmud_two_claimant_contested_garment():
    """Two hold a garment (Baba Metzia 2a): claims 1 and 1/2 -> 3/4 and 1/4."""
    assert talmud(1.0, (1.0, 0.5)) == pytest.approx((0.75, 0.25))
    assert adjusted_proportional(1.0, (1.0, 0.5)) == pytest.approx((0.75, 0.25))


def test_talmud_branches():
    c = np.array(CLAIMS3)
    half = c / 2.0
    total = c.sum()
    for estate in (0.0, 50.0, 150.0, 300.0):
        assert talmud(estate, CLAIMS3) == pytest.approx(constrained_equal_awards(estate, half))
    for estate in (300.0, 350.0, 450.0, 600.0):
        expected = half + np.array(constrained_equal_losses(estate - total / 2, half))
        assert talmud(estate, CLAIMS3) == pytest.approx(expected)
    # at E = C/2 everybody gets exactly half the claim
    assert talmud(total / 2, CLAIMS3) == pytest.approx(tuple(half))


def test_talmud_half_claim_property():
    """Nobody gets more than half their claim iff E <= C/2."""
    c = np.array(CLAIMS3)
    for estate in np.linspace(0.0, 600.0, 61):
        a = np.array(talmud(float(estate), CLAIMS3))
        if estate <= 300.0:
            assert np.all(a <= c / 2 + 1e-12)
        else:
            assert np.all(a >= c / 2 - 1e-12)


def test_talmud_self_duality():
    """T(E, c) = c - T(C - E, c)."""
    rng = np.random.default_rng(9)
    for _ in range(50):
        claims = rng.uniform(0.0, 30.0, int(rng.integers(1, 7)))
        total = claims.sum()
        estate = float(rng.uniform(0.0, total))
        lhs = np.array(talmud(estate, claims))
        rhs = claims - np.array(talmud(total - estate, claims))
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_talmud_equals_adjusted_proportional_for_two_claimants():
    rng = np.random.default_rng(21)
    for _ in range(50):
        claims = rng.uniform(0.0, 10.0, 2)
        estate = float(rng.uniform(0.0, claims.sum()))
        assert talmud(estate, claims) == pytest.approx(adjusted_proportional(estate, claims), abs=1e-9)


# --------------------------------------------------------------------------- #
# Adjusted proportional (Curiel, Maschler & Tijs 1987) and minimal rights
# --------------------------------------------------------------------------- #
def test_minimal_rights_closed_form():
    assert minimal_rights(400.0, CLAIMS3) == pytest.approx((0.0, 0.0, 100.0))
    assert minimal_rights(550.0, CLAIMS3) == pytest.approx((50.0, 150.0, 250.0))
    assert minimal_rights(300.0, CLAIMS3) == (0.0, 0.0, 0.0)
    assert minimal_rights(0.0, CLAIMS3) == (0.0, 0.0, 0.0)
    assert minimal_rights(1e9, CLAIMS3) == pytest.approx(CLAIMS3)
    assert minimal_rights(float("inf"), CLAIMS3_DICT) == pytest.approx(CLAIMS3_DICT)
    assert minimal_rights(5.0, []) == []


def test_minimal_rights_sum_never_exceeds_estate():
    rng = np.random.default_rng(2)
    for _ in range(50):
        claims = rng.uniform(0.0, 10.0, int(rng.integers(1, 6)))
        estate = float(rng.uniform(0.0, 1.2 * claims.sum()))
        m = np.array(minimal_rights(estate, claims))
        assert m.sum() <= min(estate, claims.sum()) + 1e-9
        assert np.all(m <= claims + 1e-12)


def test_adjusted_proportional_closed_form():
    # m = (0, 0, 100); E' = 300; c' = (100, 200, 200); P -> (60, 120, 120)
    assert adjusted_proportional(400.0, CLAIMS3) == pytest.approx((60.0, 120.0, 220.0))
    # m = (0, 50, 150); E' = 250; c' = (100, 150, 150); P -> (62.5, 93.75, 93.75)
    assert adjusted_proportional(450.0, CLAIMS3) == pytest.approx((62.5, 143.75, 243.75))
    assert adjusted_proportional(400.0, CLAIMS3_DICT) == pytest.approx(
        {"Highland": 60.0, "Midland": 120.0, "Delta": 220.0}
    )


def test_adjusted_proportional_honours_minimal_rights():
    rng = np.random.default_rng(4)
    for _ in range(50):
        claims = rng.uniform(0.0, 10.0, int(rng.integers(1, 6)))
        estate = float(rng.uniform(0.0, claims.sum()))
        ap = np.array(adjusted_proportional(estate, claims))
        m = np.array(minimal_rights(estate, claims))
        assert np.all(ap >= m - 1e-9)


def test_adjusted_proportional_reduces_to_proportional_without_rights_or_truncation():
    # E = 300: no minimal rights (E <= C - max c) and no truncation (E >= max c)
    assert adjusted_proportional(300.0, CLAIMS3) == pytest.approx(proportional(300.0, CLAIMS3))


def test_adjusted_proportional_differs_from_proportional_in_general():
    assert adjusted_proportional(400.0, CLAIMS3) != pytest.approx(proportional(400.0, CLAIMS3))
    # small estate: claims are truncated at E, which favours the small claimant
    assert adjusted_proportional(50.0, CLAIMS3) == pytest.approx((50 / 3,) * 3)


# --------------------------------------------------------------------------- #
# Upstream priority
# --------------------------------------------------------------------------- #
def test_upstream_priority_serves_in_order():
    assert upstream_priority(250.0, CLAIMS3) == (100.0, 150.0, 0.0)
    assert upstream_priority(50.0, CLAIMS3) == (50.0, 0.0, 0.0)
    assert upstream_priority(350.0, CLAIMS3) == (100.0, 200.0, 50.0)
    assert upstream_priority(float("inf"), CLAIMS3) == CLAIMS3
    assert upstream_priority(250.0, CLAIMS3_DICT) == {"Highland": 100.0, "Midland": 150.0, "Delta": 0.0}


def test_upstream_priority_is_not_symmetric():
    assert upstream_priority(30.0, (20.0, 20.0, 20.0)) == (20.0, 10.0, 0.0)
    a = upstream_priority(250.0, CLAIMS3)
    b = upstream_priority(250.0, CLAIMS3[::-1])
    assert a != b[::-1]


# --------------------------------------------------------------------------- #
# compare_rules
# --------------------------------------------------------------------------- #
def test_compare_rules_default_and_subset():
    table = compare_rules(400.0, CLAIMS3_DICT)
    assert list(table) == RULE_NAMES
    for name, awards in table.items():
        assert isinstance(awards, dict)
        assert list(awards) == list(CLAIMS3_DICT)
        assert awards == apply_rule(name, 400.0, CLAIMS3_DICT)
    sub = compare_rules(400.0, CLAIMS3_DICT, rules=["Talmud", "adjusted_proportional"])
    assert list(sub) == ["talmud", "ap"]
    assert sub["talmud"] == pytest.approx({"Highland": 50.0, "Midland": 125.0, "Delta": 225.0})
    assert isinstance(compare_rules(400.0, CLAIMS3)["cea"], tuple)


def test_compare_rules_validation():
    with pytest.raises(ValueError):
        compare_rules(400.0, CLAIMS3_DICT, rules=["talmud", "nope"])
    with pytest.raises(ValueError):
        compare_rules(400.0, CLAIMS3_DICT, rules=["talmud", "contested_garment"])  # duplicate
    with pytest.raises(ValueError):
        compare_rules(-1.0, CLAIMS3_DICT)
    with pytest.raises(ValueError):
        compare_rules(400.0, {"a": -1.0})


# --------------------------------------------------------------------------- #
# Bankruptcy game (O'Neill 1982) links the rules to the game solutions
# --------------------------------------------------------------------------- #
def test_bankruptcy_game_values():
    v = bankruptcy_game(400.0, CLAIMS3_DICT)
    assert v(frozenset()) == 0.0
    assert v(frozenset(["Highland"])) == 0.0
    assert v(frozenset(["Delta"])) == 100.0
    assert v(frozenset(["Midland", "Delta"])) == 300.0
    assert v(frozenset(CLAIMS3_DICT)) == 400.0
    assert v("Delta") == 100.0  # a bare name is a singleton
    with pytest.raises(ValueError):
        v(frozenset(["Nobody"]))
    # sequence claims -> integer players
    w = bankruptcy_game(400.0, CLAIMS3)
    assert w(frozenset([2])) == 100.0
    assert w(frozenset([0, 1, 2])) == 400.0
    with pytest.raises(ValueError):
        bankruptcy_game(-1.0, CLAIMS3)
    with pytest.raises(ValueError):
        bankruptcy_game(float("inf"), CLAIMS3)


def test_bankruptcy_game_singletons_are_minimal_rights_and_game_is_convex():
    v = bankruptcy_game(400.0, CLAIMS3_DICT)
    m = minimal_rights(400.0, CLAIMS3_DICT)
    for name in CLAIMS3_DICT:
        assert v(frozenset([name])) == pytest.approx(m[name])
    names = list(CLAIMS3_DICT)
    subsets = [frozenset(names[i] for i in range(3) if (mask >> i) & 1) for mask in range(8)]
    for s in subsets:
        for t in subsets:
            assert v(s | t) + v(s & t) >= v(s) + v(t) - 1e-12


@pytest.mark.parametrize("estate", [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 750.0, 1e4])
def test_nucleolus_of_bankruptcy_game_is_talmud(estate):
    """Aumann & Maschler (1985): the Talmud rule is the nucleolus of O'Neill's game."""
    v = bankruptcy_game(estate, CLAIMS3_DICT)
    nuc = nucleolus(list(CLAIMS3_DICT), v)
    assert nuc == pytest.approx(talmud(estate, CLAIMS3_DICT), abs=1e-6)


def test_nucleolus_of_bankruptcy_game_is_talmud_five_claimants():
    claims = {"a": 3.0, "b": 7.0, "c": 11.0, "d": 20.0, "e": 44.0}
    for estate in (10.0, 40.0, 42.5, 60.0, 80.0):
        v = bankruptcy_game(estate, claims)
        assert nucleolus(list(claims), v) == pytest.approx(talmud(estate, claims), abs=1e-6)


def test_shapley_of_two_claimant_bankruptcy_game_is_contested_garment():
    claims = {"a": 100.0, "b": 300.0}
    v = bankruptcy_game(200.0, claims)
    assert shapley_value(list(claims), v) == pytest.approx(talmud(200.0, claims))
    assert shapley_value(list(claims), v) == pytest.approx({"a": 50.0, "b": 150.0})


def test_shapley_and_talmud_lie_in_core_of_bankruptcy_game():
    names = list(CLAIMS3_DICT)
    for estate in (100.0, 250.0, 400.0, 550.0, 600.0, 900.0):
        v = bankruptcy_game(estate, CLAIMS3_DICT)
        assert is_in_core(shapley_value(names, v), names, v, tol=1e-9)
        assert is_in_core(talmud(estate, CLAIMS3_DICT), names, v, tol=1e-9)


def test_bankruptcy_game_estate_above_total_claims_review_example():
    """E > C (a wet year) must give v(empty) == 0 and v(N) == C, not raise in the solvers."""
    v = bankruptcy_game(10.0, [1.0, 2.0])
    assert v(frozenset()) == 0.0
    assert v(frozenset([0])) == 1.0
    assert v(frozenset([1])) == 2.0
    assert v(frozenset([0, 1])) == 3.0  # min(E, C), not E
    claims = {0: 1.0, 1: 2.0}
    assert shapley_value([0, 1], v) == pytest.approx(claims)
    assert nucleolus([0, 1], v) == pytest.approx(claims, abs=1e-7)
    assert talmud(10.0, [1.0, 2.0]) == [1.0, 2.0]
    assert is_in_core(claims, [0, 1], v)
    violations = core_constraints_violations(claims, [0, 1], v)
    assert set(violations) == {frozenset([0]), frozenset([1]), frozenset([0, 1])}
    assert all(abs(x) <= 1e-12 for x in violations.values())


@pytest.mark.parametrize("estate", [600.0, 600.5, 1e4, 1e9])
def test_bankruptcy_game_is_additive_when_estate_covers_claims(estate):
    """For E >= C the truncated game is additive and every solution returns the claims."""
    names = list(CLAIMS3_DICT)
    v = bankruptcy_game(estate, CLAIMS3_DICT)
    assert v(frozenset()) == 0.0
    assert v(frozenset(names)) == pytest.approx(600.0)
    for mask in range(8):
        s = frozenset(names[i] for i in range(3) if (mask >> i) & 1)
        assert v(s) == pytest.approx(sum(CLAIMS3_DICT[p] for p in s))
    assert shapley_value(names, v) == pytest.approx(CLAIMS3_DICT)
    assert nucleolus(names, v) == pytest.approx(CLAIMS3_DICT, abs=1e-6)
    assert nucleolus(names, v) == pytest.approx(talmud(estate, CLAIMS3_DICT), abs=1e-6)
    assert nucleolus(names, v) == pytest.approx(apply_rule("ap", estate, CLAIMS3_DICT), abs=1e-6)
    assert is_in_core(CLAIMS3_DICT, names, v)
    # the sequence form behaves the same with integer players
    w = bankruptcy_game(estate, CLAIMS3)
    assert w(frozenset([0, 1, 2])) == pytest.approx(600.0)
    assert shapley_value([0, 1, 2], w) == pytest.approx(dict(enumerate(CLAIMS3)))


@pytest.mark.parametrize("estate", [0.0, 1.0, 99.9, 300.0, 599.9, 600.0, 600.1, 1e6])
def test_bankruptcy_game_grand_coalition_and_bounds(estate):
    """v(empty) == 0, v(N) == min(E, C), 0 <= v(S) <= c(S), singletons are minimal rights, convex."""
    names = list(CLAIMS3_DICT)
    v = bankruptcy_game(estate, CLAIMS3_DICT)
    m = minimal_rights(estate, CLAIMS3_DICT)
    subsets = [frozenset(names[i] for i in range(3) if (mask >> i) & 1) for mask in range(8)]
    assert v(frozenset()) == 0.0
    assert v(frozenset(names)) == pytest.approx(min(estate, 600.0))
    for s in subsets:
        assert 0.0 <= v(s) <= sum(CLAIMS3_DICT[p] for p in s) + 1e-12
    for p in names:
        assert v(frozenset([p])) == pytest.approx(m[p])
    for s in subsets:
        for t in subsets:
            assert v(s | t) + v(s & t) >= v(s) + v(t) - 1e-12


def test_bankruptcy_game_worth_is_monotone_in_estate_and_saturates():
    names = list(CLAIMS3_DICT)
    estates = [0.0, 50.0, 150.0, 400.0, 599.0, 600.0, 601.0, 5e3]
    subsets = [frozenset(names[i] for i in range(3) if (mask >> i) & 1) for mask in range(8)]
    for s in subsets:
        worths = [bankruptcy_game(e, CLAIMS3_DICT)(s) for e in estates]
        assert all(b >= a - 1e-12 for a, b in zip(worths, worths[1:]))
        assert worths[-1] == pytest.approx(worths[-3])  # E >= C: saturated at c(S)


# --------------------------------------------------------------------------- #
# game_from_dict
# --------------------------------------------------------------------------- #
def test_game_from_dict_keys_and_defaults():
    v = game_from_dict({("a", "b"): 5, "a": 1, frozenset(["b"]): 2})
    assert v(frozenset()) == 0.0
    assert v(frozenset(["a"])) == 1.0
    assert v("b") == 2.0
    assert v(frozenset(["a", "b"])) == 5.0
    with pytest.raises(ValueError):
        v(frozenset(["c"]))
    w = game_from_dict({("a", "b"): 5}, default=0.0)
    assert w(frozenset(["a"])) == 0.0
    assert w(frozenset(["z", "y"])) == 0.0
    with pytest.raises(ValueError):
        game_from_dict([("a", 1)])
    with pytest.raises(ValueError):
        game_from_dict({"a": "one"})
    # booleans are allowed (simple games)
    assert game_from_dict({"a": True})("a") == 1.0


# --------------------------------------------------------------------------- #
# Shapley value
# --------------------------------------------------------------------------- #
def _glove(S):
    return min(len(S & {"L"}), len(S & {"R1", "R2"}))


def _majority(S):
    return 1.0 if len(S) >= 2 else 0.0


def _airport(S):
    costs = {"small": 3.0, "medium": 6.0, "large": 10.0}
    return max((costs[p] for p in S), default=0.0)


def test_shapley_three_player_majority_game():
    phi = shapley_value(["a", "b", "c"], _majority)
    assert phi == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert sum(phi.values()) == pytest.approx(1.0)


def test_shapley_glove_game():
    """One left glove, two right gloves: (2/3, 1/6, 1/6)."""
    phi = shapley_value(["L", "R1", "R2"], _glove)
    assert phi == pytest.approx({"L": 2 / 3, "R1": 1 / 6, "R2": 1 / 6})


def test_shapley_airport_game_littlechild_owen():
    """Runway costs 3, 6, 10 -> 1, 2.5, 6.5 (Littlechild & Owen 1973)."""
    phi = shapley_value(["small", "medium", "large"], _airport)
    assert phi == pytest.approx({"small": 1.0, "medium": 2.5, "large": 6.5})


def test_shapley_efficiency_random_games():
    rng = np.random.default_rng(8)
    for n in (1, 2, 3, 4, 5, 6):
        names = [f"p{i}" for i in range(n)]
        table = {frozenset(): 0.0}
        for mask in range(1, 1 << n):
            table[frozenset(names[i] for i in range(n) if (mask >> i) & 1)] = float(rng.normal())
        v = game_from_dict(table)
        phi = shapley_value(names, v)
        assert list(phi) == names
        assert sum(phi.values()) == pytest.approx(v(frozenset(names)), abs=1e-9)


def test_shapley_symmetry_dummy_and_additivity():
    def v(S):
        s = S & {"a", "b"}
        return (4.0 if len(s) == 2 else 1.0 * len(s)) + (2.0 if "d" in S else 0.0)

    phi = shapley_value(["a", "b", "c", "d"], v)
    assert phi["a"] == pytest.approx(phi["b"])  # symmetric players
    assert phi["c"] == pytest.approx(0.0)  # null player
    assert phi["d"] == pytest.approx(2.0)  # dummy player gets v({d})
    assert sum(phi.values()) == pytest.approx(6.0)

    def w(S):
        return len(S) ** 2

    both = shapley_value(["a", "b", "c", "d"], lambda S: v(S) + w(S))
    phi_w = shapley_value(["a", "b", "c", "d"], w)
    for p in both:
        assert both[p] == pytest.approx(phi[p] + phi_w[p])


def test_shapley_unanimity_game_splits_equally_among_carrier():
    carrier = {"x", "y"}
    phi = shapley_value(["x", "y", "z"], lambda S: 1.0 if carrier <= S else 0.0)
    assert phi == pytest.approx({"x": 0.5, "y": 0.5, "z": 0.0})


def test_shapley_convex_game_in_core():
    weights = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 0.5}
    v = lambda S: sum(weights[p] for p in S) ** 2  # noqa: E731 - supermodular
    names = list(weights)
    assert is_in_core(shapley_value(names, v), names, v, tol=1e-9)


def test_shapley_accepts_boolean_games_and_integer_players():
    phi = shapley_value((1, 2, 3), lambda S: len(S) >= 2)
    assert phi == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    assert shapley_value(["only"], lambda S: 7.0 if S else 0.0) == {"only": 7.0}


def test_shapley_does_not_mutate_players():
    players = ["a", "b", "c"]
    snapshot = list(players)
    shapley_value(players, _majority)
    assert players == snapshot


@pytest.mark.parametrize(
    "players, v",
    [
        ([], _majority),
        (["a", "a"], _majority),
        ("abc", _majority),
        (["a", "b"], 3.0),
        (["a", "b"], lambda S: 1.0),  # v(empty) != 0
        (["a", "b"], lambda S: None),
        (["a", "b"], lambda S: "x"),
        (["a", "b"], lambda S: float("nan")),
        (["a", "b"], lambda S: float("inf")),
    ],
)
def test_shapley_validation(players, v):
    with pytest.raises(ValueError):
        shapley_value(players, v)


def test_shapley_max_players_guard():
    players = [f"p{i}" for i in range(5)]
    with pytest.raises(ValueError, match="max_players"):
        shapley_value(players, _majority, max_players=4)
    assert sum(shapley_value(players, _majority, max_players=5).values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def test_core_of_glove_game_is_a_single_point():
    names = ["L", "R1", "R2"]
    assert is_in_core({"L": 1.0, "R1": 0.0, "R2": 0.0}, names, _glove)
    assert not is_in_core(shapley_value(names, _glove), names, _glove)
    assert not is_in_core({"L": 0.9, "R1": 0.05, "R2": 0.05}, names, _glove)


def test_core_of_majority_game_is_empty():
    names = ["a", "b", "c"]
    for x in ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {"a": 0.5, "b": 0.5, "c": 0.0}, {"a": 1.0, "b": 0.0, "c": 0.0}):
        assert not is_in_core(x, names, _majority)
        viol = core_constraints_violations(x, names, _majority)
        assert max(viol.values()) > 1e-9


def test_core_constraints_violations_structure():
    names = ["L", "R1", "R2"]
    x = shapley_value(names, _glove)
    viol = core_constraints_violations(x, names, _glove)
    assert len(viol) == 2 ** 3 - 1
    assert all(isinstance(k, frozenset) for k in viol)
    assert viol[frozenset(["L", "R1"])] == pytest.approx(1.0 - (2 / 3 + 1 / 6))
    assert viol[frozenset(["R1", "R2"])] == pytest.approx(-(1 / 3))
    assert viol[frozenset(names)] == pytest.approx(0.0)
    assert viol[frozenset(["L"])] == pytest.approx(-2 / 3)


def test_is_in_core_tolerance_and_efficiency():
    names = ["L", "R1", "R2"]
    assert is_in_core({"L": 1.0 - 5e-10, "R1": 2.5e-10, "R2": 2.5e-10}, names, _glove)
    assert not is_in_core({"L": 1.0 - 5e-10, "R1": 2.5e-10, "R2": 2.5e-10}, names, _glove, tol=1e-12)
    assert not is_in_core({"L": 1.5, "R1": 0.0, "R2": 0.0}, names, _glove)  # over-distributes
    assert not is_in_core({"L": 0.5, "R1": 0.0, "R2": 0.0}, names, _glove)  # under-distributes


def test_core_validation():
    names = ["a", "b"]
    v = game_from_dict({"a": 0, "b": 0, ("a", "b"): 1})
    with pytest.raises(ValueError):
        is_in_core({"a": 0.5}, names, v)
    with pytest.raises(ValueError):
        is_in_core({"a": 0.5, "b": 0.5, "c": 0.0}, names, v)
    with pytest.raises(ValueError):
        is_in_core([0.5, 0.5], names, v)
    with pytest.raises(ValueError):
        is_in_core({"a": 0.5, "b": 0.5}, names, v, tol=-1.0)
    with pytest.raises(ValueError):
        core_constraints_violations({"a": 0.5, "b": "x"}, names, v)


# --------------------------------------------------------------------------- #
# Nucleolus
# --------------------------------------------------------------------------- #
def test_nucleolus_two_player_standard_solution():
    v = game_from_dict({"1": 2.0, "2": 5.0, ("1", "2"): 11.0})
    nuc = nucleolus(["1", "2"], v)
    assert nuc == pytest.approx({"1": 4.0, "2": 7.0}, abs=1e-8)
    assert nuc == pytest.approx(shapley_value(["1", "2"], v), abs=1e-8)


def test_nucleolus_symmetric_games_equal_shapley():
    names = ["a", "b", "c"]
    assert nucleolus(names, _majority) == pytest.approx(shapley_value(names, _majority), abs=1e-8)
    names4 = ["w", "x", "y", "z"]
    v = lambda S: float(len(S) ** 2)  # noqa: E731
    nuc = nucleolus(names4, v)
    assert nuc == pytest.approx(shapley_value(names4, v), abs=1e-8)
    assert nuc == pytest.approx({p: 4.0 for p in names4}, abs=1e-8)


def test_nucleolus_glove_game_is_the_unique_core_point():
    names = ["L", "R1", "R2"]
    nuc = nucleolus(names, _glove)
    assert nuc == pytest.approx({"L": 1.0, "R1": 0.0, "R2": 0.0}, abs=1e-8)
    assert is_in_core(nuc, names, _glove, tol=1e-7)


def test_nucleolus_in_core_of_convex_game():
    rng = np.random.default_rng(11)
    for n in (2, 3, 4, 5):
        weights = rng.uniform(0.2, 3.0, n)
        names = [f"r{i}" for i in range(n)]
        v = lambda S, w=weights, nm=names: sum(w[nm.index(p)] for p in S) ** 2  # noqa: E731
        nuc = nucleolus(names, v)
        assert list(nuc) == names
        assert sum(nuc.values()) == pytest.approx(v(frozenset(names)), abs=1e-7)
        assert is_in_core(nuc, names, v, tol=1e-6)


def test_nucleolus_two_level_hand_example():
    """v(12) = 4, v(N) = 10, all else 0: level 1 fixes x3 = 3, level 2 splits 7 equally."""
    v = game_from_dict({("1", "2"): 4.0, ("1", "2", "3"): 10.0}, default=0.0)
    nuc = nucleolus(["1", "2", "3"], v)
    assert nuc == pytest.approx({"1": 3.5, "2": 3.5, "3": 3.0}, abs=1e-8)
    assert shapley_value(["1", "2", "3"], v) == pytest.approx({"1": 4.0, "2": 4.0, "3": 2.0})
    assert is_in_core(nuc, ["1", "2", "3"], v, tol=1e-7)


def test_nucleolus_scale_invariance():
    v = game_from_dict({("1", "2"): 4.0, ("1", "2", "3"): 10.0}, default=0.0)
    big = nucleolus(["1", "2", "3"], lambda S: 1e6 * v(S))
    small = nucleolus(["1", "2", "3"], lambda S: 1e-4 * v(S))
    assert big == pytest.approx({"1": 3.5e6, "2": 3.5e6, "3": 3.0e6}, rel=1e-8)
    assert small == pytest.approx({"1": 3.5e-4, "2": 3.5e-4, "3": 3.0e-4}, rel=1e-8)


def test_nucleolus_single_player_and_imputation():
    assert nucleolus(["solo"], lambda S: 5.0 if S else 0.0) == {"solo": 5.0}
    v = game_from_dict({"a": 1.0, "b": 2.0, "c": 0.0, ("a", "b", "c"): 6.0}, default=0.0)
    nuc = nucleolus(["a", "b", "c"], v)
    assert nuc["a"] >= 1.0 - 1e-9 and nuc["b"] >= 2.0 - 1e-9 and nuc["c"] >= -1e-9
    assert sum(nuc.values()) == pytest.approx(6.0)


def test_nucleolus_inessential_game_and_prenucleolus():
    v = game_from_dict({"1": 1.0, "2": 1.0, ("1", "2"): 0.0})
    with pytest.raises(ValueError, match="essential"):
        nucleolus(["1", "2"], v)
    pre = nucleolus(["1", "2"], v, imputation=False)
    assert pre == pytest.approx({"1": 0.0, "2": 0.0}, abs=1e-8)


def test_nucleolus_player_limit_and_validation():
    players = [f"p{i}" for i in range(9)]
    with pytest.raises(ValueError, match="max_players"):
        nucleolus(players, _majority)
    with pytest.raises(ValueError, match="max_players"):
        nucleolus(["a", "b", "c"], _majority, max_players=2)
    with pytest.raises(ValueError):
        nucleolus(["a", "a"], _majority)
    with pytest.raises(ValueError):
        nucleolus(["a", "b"], lambda S: 1.0)  # v(empty) != 0
    with pytest.raises(ValueError):
        nucleolus(["a", "b"], _majority, tol=-1.0)


def test_nucleolus_does_not_mutate_players():
    players = ["L", "R1", "R2"]
    snapshot = list(players)
    nucleolus(players, _glove)
    assert players == snapshot


# --------------------------------------------------------------------------- #
# Nash bargaining
# --------------------------------------------------------------------------- #
def test_nash_symmetric_two_players_split_equally():
    sol = nash_bargaining({"up": lambda x: x, "down": lambda x: x}, {"up": 0.0, "down": 0.0}, 100.0)
    assert sol == pytest.approx({"up": 50.0, "down": 50.0})
    sol = nash_bargaining({"up": math.sqrt, "down": math.sqrt}, {"up": 0.0, "down": 0.0}, 100.0)
    assert sol == pytest.approx({"up": 50.0, "down": 50.0})


def test_nash_scale_invariance_of_utilities():
    base = nash_bargaining({"a": lambda x: x, "b": lambda x: x}, {"a": 10.0, "b": 0.0}, 100.0)
    scaled = nash_bargaining({"a": lambda x: 3.0 * x + 7.0, "b": lambda x: 0.5 * x}, {"a": 37.0, "b": 0.0}, 100.0)
    assert scaled == pytest.approx(base, abs=1e-6)


def test_nash_better_disagreement_point_gets_more():
    sol = nash_bargaining({"a": lambda x: x, "b": lambda x: x}, {"a": 30.0, "b": 0.0}, 100.0)
    assert sol["a"] > sol["b"]
    # linear utilities: split the surplus over the disagreement payoffs equally
    assert sol == pytest.approx({"a": 65.0, "b": 35.0}, abs=1e-6)
    worse = nash_bargaining({"a": lambda x: x, "b": lambda x: x}, {"a": 10.0, "b": 0.0}, 100.0)
    assert worse["a"] < sol["a"]


def test_nash_two_player_nonlinear_closed_form():
    """max x (T - x)^2 -> x = T/3."""
    fns = {"a": lambda x: x, "b": lambda x: x ** 2}
    d = {"a": 0.0, "b": 0.0}
    sol = nash_bargaining(fns, d, 3.0)
    assert sol == pytest.approx({"a": 1.0, "b": 2.0}, abs=1e-6)
    coarse = nash_bargaining(fns, d, 3.0, grid=301, refine=False)
    assert coarse == pytest.approx({"a": 1.0, "b": 2.0}, abs=3.0 / 300)
    assert coarse["a"] + coarse["b"] == pytest.approx(3.0)


def test_nash_two_player_feasibility_and_edge_cases():
    fns = {"a": lambda x: x, "b": lambda x: x}
    with pytest.raises(ValueError, match="feasible"):
        nash_bargaining(fns, {"a": 80.0, "b": 80.0}, 100.0)
    assert nash_bargaining(fns, {"a": 0.0, "b": 0.0}, 0.0) == {"a": 0.0, "b": 0.0}
    # exactly feasible boundary: the disagreement utilities use up the whole total
    sol = nash_bargaining(fns, {"a": 60.0, "b": 40.0}, 100.0)
    assert sol == pytest.approx({"a": 60.0, "b": 40.0}, abs=1e-6)
    sol = nash_bargaining(fns, {"a": 0.0, "b": 0.0}, 10.0, grid=3)
    assert sol == pytest.approx({"a": 5.0, "b": 5.0})


def test_nash_three_players_linear_closed_form():
    fns = {"a": lambda x: x, "b": lambda x: 2.0 * x, "c": lambda x: 0.5 * x}
    sol = nash_bargaining(fns, {"a": 0.0, "b": 0.0, "c": 0.0}, 90.0)
    assert sol == pytest.approx({"a": 30.0, "b": 30.0, "c": 30.0}, rel=1e-5)
    fns = {"a": lambda x: x, "b": lambda x: x, "c": lambda x: x}
    sol = nash_bargaining(fns, {"a": 30.0, "b": 0.0, "c": 15.0}, 90.0)
    assert sol == pytest.approx({"a": 45.0, "b": 15.0, "c": 30.0}, rel=1e-5)
    assert sum(sol.values()) == pytest.approx(90.0)
    assert sol["a"] > sol["c"] > sol["b"]


def test_nash_n_players_nonlinear_closed_form_and_symmetry():
    # max x y^2 z s.t. x + y + z = 4 -> (1, 2, 1)
    sol = nash_bargaining({"x": lambda t: t, "y": lambda t: t ** 2, "z": lambda t: t}, {"x": 0, "y": 0, "z": 0}, 4.0)
    assert sol == pytest.approx({"x": 1.0, "y": 2.0, "z": 1.0}, rel=1e-5)
    sym = nash_bargaining({k: math.sqrt for k in "abcde"}, {k: 0.0 for k in "abcde"}, 10.0)
    assert sym == pytest.approx({k: 2.0 for k in "abcde"}, rel=1e-5)


def test_nash_n_players_feasibility_and_edge_cases():
    fns = {"a": lambda x: x, "b": lambda x: x, "c": lambda x: x}
    with pytest.raises(ValueError, match="feasible"):
        nash_bargaining(fns, {"a": 50.0, "b": 50.0, "c": 50.0}, 100.0)
    assert nash_bargaining(fns, {"a": 0.0, "b": 0.0, "c": 0.0}, 0.0) == {"a": 0.0, "b": 0.0, "c": 0.0}
    with pytest.raises(ValueError, match="feasible"):
        nash_bargaining(fns, {"a": 1.0, "b": 0.0, "c": 0.0}, 0.0)
    assert sum(nash_bargaining(fns, {"a": 10.0, "b": 20.0, "c": 30.0}, 100.0).values()) == pytest.approx(100.0)


def test_nash_single_player():
    assert nash_bargaining({"a": lambda x: x}, {"a": 0.0}, 42.0) == {"a": 42.0}
    with pytest.raises(ValueError, match="feasible"):
        nash_bargaining({"a": lambda x: x}, {"a": 50.0}, 42.0)


@pytest.mark.parametrize(
    "fns, d, total, kwargs",
    [
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0}, 10.0, {}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0, "c": 0.0}, 10.0, {}),
        ({"a": lambda x: x, "b": 2.0}, {"a": 0.0, "b": 0.0}, 10.0, {}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": float("nan")}, 10.0, {}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, -1.0, {}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, float("inf"), {}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, 10.0, {"grid": 2}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, 10.0, {"grid": True}),
        ({"a": lambda x: x, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, 10.0, {"grid": 2.5}),
        ({"a": lambda x: None, "b": lambda x: x}, {"a": 0.0, "b": 0.0}, 10.0, {}),
        ({"a": lambda x: float("nan"), "b": lambda x: x}, {"a": 0.0, "b": 0.0}, 10.0, {}),
        ({}, {}, 10.0, {}),
        ([lambda x: x], {"a": 0.0}, 10.0, {}),
        ({"a": lambda x: x, "b": lambda x: x}, [0.0, 0.0], 10.0, {}),
    ],
)
def test_nash_validation(fns, d, total, kwargs):
    with pytest.raises(ValueError):
        nash_bargaining(fns, d, total, **kwargs)


def test_nash_does_not_mutate_inputs():
    fns = {"a": lambda x: x, "b": lambda x: x, "c": lambda x: x}
    d = {"a": 1.0, "b": 2.0, "c": 3.0}
    fns_snapshot, d_snapshot = dict(fns), dict(d)
    nash_bargaining(fns, d, 50.0)
    nash_bargaining({"a": fns["a"], "b": fns["b"]}, {"a": 1.0, "b": 2.0}, 50.0)
    assert fns == fns_snapshot and d == d_snapshot


# --------------------------------------------------------------------------- #
# Gini
# --------------------------------------------------------------------------- #
def test_gini_known_values():
    assert gini([1, 1, 1, 1]) == 0.0
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)
    assert gini([1, 0]) == pytest.approx(0.5)
    assert gini([0, 1, 1]) == pytest.approx(1 / 3)
    assert gini({"a": 2.0, "b": 4.0}) == pytest.approx(1 / 6)
    assert gini((5.0,)) == 0.0
    assert gini([]) == 0.0
    assert gini([0.0, 0.0]) == 0.0
    assert gini(np.array([3.0, 3.0, 3.0])) == 0.0


def test_gini_properties():
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = int(rng.integers(2, 10))
        x = rng.uniform(0.0, 10.0, n)
        g = gini(x)
        assert 0.0 <= g <= 1.0 - 1.0 / n + 1e-12
        assert gini(x * 7.5) == pytest.approx(g)  # scale invariant
        assert gini(x + 5.0) <= g + 1e-12  # equal transfers reduce inequality
        assert gini(x[::-1]) == pytest.approx(g)  # anonymous
        # explicit mean-absolute-difference definition
        mad = np.abs(x[:, None] - x[None, :]).sum() / (2 * n * n * x.mean())
        assert g == pytest.approx(mad)


@pytest.mark.parametrize("bad", [[1.0, -1.0], [float("nan"), 1.0], [float("inf")], ["a"], [True], "123", 5.0])
def test_gini_validation(bad):
    with pytest.raises(ValueError):
        gini(bad)


# --------------------------------------------------------------------------- #
# Satisfaction, equal satisfaction (contract ``envy_free``) and Foley no-envy
# --------------------------------------------------------------------------- #
def test_satisfaction_values():
    assert satisfaction({"a": 50.0, "b": 0.0, "c": 300.0}, {"a": 100.0, "b": 0.0, "c": 150.0}) == {
        "a": 0.5,
        "b": 1.0,
        "c": 2.0,
    }
    assert satisfaction([1.0, 2.0], [2.0, 2.0]) == [0.5, 1.0]
    assert satisfaction((1.0,), (4.0,)) == (0.25,)
    assert satisfaction([], []) == []
    # key order follows claims, not awards
    out = satisfaction({"b": 1.0, "a": 2.0}, {"a": 4.0, "b": 4.0})
    assert list(out) == ["a", "b"] and out == {"a": 0.5, "b": 0.25}


def test_satisfaction_validation():
    with pytest.raises(ValueError):
        satisfaction({"a": 1.0}, {"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        satisfaction([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        satisfaction([1.0, 2.0], {"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError):
        satisfaction({"a": -1.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        satisfaction({"a": 1.0}, {"a": float("nan")})


@pytest.mark.parametrize("fn", [envy_free, equal_satisfaction])
def test_equal_satisfaction_contract_behaviour(fn):
    """``envy_free`` is the contract name of :func:`equal_satisfaction` (satisfaction spread <= tol)."""
    claims = {"a": 100.0, "b": 200.0, "c": 300.0}
    assert fn(proportional(250.0, claims), claims)
    assert not fn(constrained_equal_awards(250.0, claims), claims)
    assert not fn(talmud(250.0, claims), claims)
    assert fn(talmud(300.0, claims), claims)  # everybody gets half
    assert fn({}, {})
    assert fn([], [])
    assert fn([5.0], [10.0])
    assert fn([1.0, 2.0], [2.0, 4.0])
    assert fn((1.0, 2.0), (2.0, 4.0))
    assert fn({"a": 1.0, "b": 1.0 + 1e-10}, {"a": 2.0, "b": 2.0})
    assert not fn({"a": 1.0, "b": 1.0 + 1e-10}, {"a": 2.0, "b": 2.0}, tol=0.0)
    assert fn({"a": 0.0, "b": 10.0}, {"a": 0.0, "b": 10.0})  # zero claim counts as satisfied
    assert not fn({"a": 0.0, "b": 5.0}, {"a": 0.0, "b": 10.0})  # ... so a half-served claimant is below it
    assert fn({"b": 100.0, "a": 50.0}, {"a": 100.0, "b": 200.0})  # award key order is irrelevant
    with pytest.raises(ValueError):
        fn({"a": 1.0}, {"a": 1.0}, tol=-1e-3)
    with pytest.raises(ValueError):
        fn({"a": 1.0}, {"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        fn([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        fn([1.0, 2.0], {"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError):
        fn({"a": -1.0}, {"a": 1.0})


def test_envy_free_is_equal_satisfaction():
    rng = np.random.default_rng(11)
    for _ in range(100):
        n = int(rng.integers(0, 6))
        c = rng.uniform(0.0, 10.0, n)
        a = c * rng.uniform(0.0, 1.5, n)
        if rng.uniform() < 0.3:
            a = c * rng.uniform(0.0, 1.5)  # proportional -> equal satisfaction
        for tol in (0.0, 1e-9, 0.1):
            assert envy_free(list(a), list(c), tol=tol) == equal_satisfaction(list(a), list(c), tol=tol)
            expected = n == 0 or (max(satisfaction(list(a), list(c))) - min(satisfaction(list(a), list(c))) <= tol)
            assert equal_satisfaction(list(a), list(c), tol=tol) == expected


@pytest.mark.parametrize("estate, claims", PROBLEMS)
def test_proportional_rule_always_has_equal_satisfaction(estate, claims):
    awards = proportional(estate, claims)
    c_vals = list(claims.values()) if isinstance(claims, dict) else list(claims)
    a_vals = list(awards.values()) if isinstance(awards, dict) else list(awards)
    if all(ci > 0.0 for ci in c_vals):
        assert equal_satisfaction(awards, claims)
        assert envy_free(awards, claims)
    else:
        # a zero claim counts as fully served (ratio 1), so the spread test only
        # holds among the positive claims, which all receive the same share
        positive = [(ai, ci) for ai, ci in zip(a_vals, c_vals) if ci > 0.0]
        assert equal_satisfaction([ai for ai, _ in positive], [ci for _, ci in positive])
        assert equal_satisfaction(awards, claims) == (estate >= sum(c_vals) or not positive)


@pytest.mark.parametrize("rule", CLAIMS_RULES)
def test_claims_rules_have_equal_satisfaction_for_equal_claims(rule):
    assert equal_satisfaction(apply_rule(rule, 150.0, (100.0, 100.0, 100.0)), (100.0, 100.0, 100.0))
    assert not equal_satisfaction(upstream_priority(150.0, (100.0, 100.0, 100.0)), (100.0, 100.0, 100.0))


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_every_rule_has_equal_satisfaction_when_estate_covers_claims(rule):
    assert equal_satisfaction(apply_rule(rule, 600.0, CLAIMS3), CLAIMS3)
    assert equal_satisfaction(apply_rule(rule, 1e4, CLAIMS3_DICT), CLAIMS3_DICT)


def test_no_envy_differs_from_equal_satisfaction_review_examples():
    """Foley (1967): bundles, not shares.  Claims (100, 200)."""
    claims = {"a": 100.0, "b": 200.0}
    prop = {"a": 50.0, "b": 100.0}  # proportional: equal shares, but a could use b's 100 in full
    assert equal_satisfaction(prop, claims) and envy_free(prop, claims)
    assert not no_envy(prop, claims)
    cea = {"a": 100.0, "b": 100.0}  # CEA: nobody wants to swap
    assert no_envy(cea, claims)
    assert not equal_satisfaction(cea, claims) and not envy_free(cea, claims)
    assert no_envy(constrained_equal_awards(200.0, claims), claims)
    assert not no_envy(constrained_equal_losses(200.0, claims), claims)  # (50, 150)
    assert no_envy(talmud(100.0, claims), claims)  # (50, 50)
    assert not no_envy(talmud(200.0, claims), claims)  # (50, 100)
    assert not no_envy(proportional(200.0, claims), claims)
    assert not no_envy(adjusted_proportional(200.0, claims), claims)
    assert not no_envy(upstream_priority(150.0, [200.0, 100.0]), [200.0, 100.0])  # (150, 0)
    assert not no_envy(upstream_priority(150.0, [100.0, 200.0]), [100.0, 200.0])  # (100, 50): b envies a's 100
    assert no_envy(upstream_priority(100.0, [50.0, 200.0]), [50.0, 200.0])  # (50, 50): a is full, b matches it


@pytest.mark.parametrize("estate, claims", PROBLEMS)
def test_cea_and_equal_split_are_always_no_envy(estate, claims):
    assert no_envy(constrained_equal_awards(estate, claims), claims)
    assert no_envy(equal_split(estate, claims), claims)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_every_rule_is_no_envy_when_estate_covers_claims(rule):
    assert no_envy(apply_rule(rule, 600.0, CLAIMS3), CLAIMS3)
    assert no_envy(apply_rule(rule, 1e4, CLAIMS3_DICT), CLAIMS3_DICT)
    assert no_envy(apply_rule(rule, 0.0, CLAIMS3), CLAIMS3)  # nobody gets anything: nothing to envy


def test_no_envy_equal_claims_requires_equal_awards_unless_fully_served():
    claims = (2.0, 2.0, 2.0)
    assert no_envy((1.5, 1.5, 1.5), claims)
    assert no_envy((2.0, 2.0, 2.0), claims)
    assert not no_envy((1.0, 2.0, 1.5), claims)
    assert not no_envy((1.0, 1.0, 1.5), claims)
    # over-abstraction above a claim never makes the over-served envy; the others still may
    assert no_envy({"a": 3.0, "b": 2.0}, {"a": 2.0, "b": 2.0})
    assert not no_envy({"a": 3.0, "b": 1.0}, {"a": 2.0, "b": 2.0})
    # a claimant is only compared with what it could use: b's 10 is worth 1 to a
    assert no_envy({"a": 1.0, "b": 10.0}, {"a": 1.0, "b": 20.0})
    assert not no_envy({"a": 0.5, "b": 10.0}, {"a": 1.0, "b": 20.0})


def test_no_envy_matches_pairwise_definition():
    """a_i >= min(a_j, c_i) for all i, j (Foley 1967 with claim truncation)."""
    rng = np.random.default_rng(7)
    seen = {True: 0, False: 0}
    for _ in range(300):
        n = int(rng.integers(1, 6))
        c = rng.uniform(0.0, 10.0, n)
        a = c * rng.uniform(0.0, 1.3, n)
        if rng.uniform() < 0.4:  # push some towards CEA-like vectors so both outcomes are exercised
            a = np.minimum(c, rng.uniform(0.0, 10.0))
        expected = all(a[i] >= min(a[j], c[i]) for i in range(n) for j in range(n))
        got = no_envy(list(a), list(c), tol=0.0)
        assert got == expected
        seen[got] += 1
        assert no_envy({f"p{i}": a[i] for i in range(n)}, {f"p{i}": c[i] for i in range(n)}, tol=0.0) == expected
    assert seen[True] > 10 and seen[False] > 10


def test_no_envy_edge_cases_tolerance_and_validation():
    assert no_envy({}, {})
    assert no_envy([], [])
    assert no_envy([5.0], [10.0])
    assert no_envy((0.0, 0.0), (1.0, 2.0))
    assert no_envy({"a": 0.0, "b": 10.0}, {"a": 0.0, "b": 10.0})  # zero claim never envies
    assert not no_envy({"a": 0.0, "b": 10.0}, {"a": 1.0, "b": 10.0})
    assert no_envy({"a": 1.0 - 1e-10, "b": 1.0}, {"a": 2.0, "b": 2.0})
    assert not no_envy({"a": 1.0 - 1e-10, "b": 1.0}, {"a": 2.0, "b": 2.0}, tol=0.0)
    assert no_envy({"a": 0.5, "b": 1.0}, {"a": 2.0, "b": 2.0}, tol=0.5)
    assert no_envy({"b": 1.0, "a": 1.0}, {"a": 2.0, "b": 2.0})  # award key order is irrelevant
    assert no_envy(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
    with pytest.raises(ValueError):
        no_envy({"a": 1.0}, {"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        no_envy([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        no_envy([1.0, 2.0], {"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError):
        no_envy({"a": -1.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        no_envy({"a": 1.0}, {"a": float("nan")})
    with pytest.raises(ValueError):
        no_envy({"a": 1.0}, {"a": 1.0}, tol=-1e-3)
    with pytest.raises(ValueError):
        no_envy("12", "34")


def test_equity_tests_do_not_mutate_inputs():
    awards, claims = {"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 2.0}
    awards_list, claims_list = [1.0, 2.0], [2.0, 2.0]
    awards_arr, claims_arr = np.array([1.0, 2.0]), np.array([2.0, 2.0])
    snap = (dict(awards), dict(claims), list(awards_list), list(claims_list), awards_arr.copy(), claims_arr.copy())
    for fn in (envy_free, equal_satisfaction, no_envy, satisfaction):
        fn(awards, claims)
        fn(awards_list, claims_list)
        fn(awards_arr, claims_arr)
    assert (awards, claims, awards_list, claims_list) == snap[:4]
    assert np.array_equal(awards_arr, snap[4]) and np.array_equal(claims_arr, snap[5])


# --------------------------------------------------------------------------- #
# Integration with the stylised basin
# --------------------------------------------------------------------------- #
def test_rules_on_example_basin_demands(basin):
    # ``dataclasses.asdict`` deep-copies the nested ``consumption_fraction`` /
    # ``value_usd_per_m3`` dictionaries, so (unlike a guard built on
    # ``Basin.copy()`` whose depth we do not want to rely on) the comparison
    # below is sensitive to in-place edits of those sector tables too.
    before = dataclasses.asdict(basin)
    claims = {r.name: r.demand.total_withdrawal() for r in basin}
    estate = basin.total_natural_flow() - sum(r.demand.environmental for r in basin)
    assert estate > 0.0 and estate < sum(claims.values())
    table = compare_rules(estate, claims)
    for name, awards in table.items():
        assert list(awards) == basin.names()
        assert sum(awards.values()) == pytest.approx(min(estate, sum(claims.values())))
        for r in basin:
            assert 0.0 <= awards[r.name] <= claims[r.name] + 1e-9
    # the Delta's large claim makes it the biggest beneficiary of CEL, the smallest of CEA
    assert table["cel"]["Delta"] >= table["cea"]["Delta"]
    assert table["cea"]["Highland"] >= table["cel"]["Highland"]
    assert envy_free(table["proportional"], claims)
    assert equal_satisfaction(table["proportional"], claims)
    # Foley no-envy: CEA passes, the proportional split leaves the Highland envying the Delta's bundle
    assert no_envy(table["cea"], claims)
    assert not no_envy(table["proportional"], claims)
    assert 0.0 <= gini(satisfaction(table["talmud"], claims)) < 1.0
    # upstream priority starves the Delta when the estate is short
    up = table["upstream_priority"]
    assert up["Highland"] == pytest.approx(claims["Highland"])
    assert up["Delta"] < claims["Delta"]
    assert dataclasses.asdict(basin) == before  # nothing was mutated
    # explicit probes of the nested sector tables the composite modules read
    assert basin.riparians[2].demand.value_usd_per_m3[Sector.MUNICIPAL] == 1.5
    assert basin.riparians[2].demand.consumption_fraction[Sector.AGRICULTURAL] == 0.6
    assert [r.position for r in basin] == [0, 1, 2]


def test_basin_snapshot_guard_detects_nested_sector_dict_mutation(basin):
    """The ``asdict`` snapshot used by the no-mutation guards is not tautological.

    A guard of the form ``snapshot = basin.copy(); ...; assert basin == snapshot``
    is blind to in-place edits of ``consumption_fraction`` / ``value_usd_per_m3``
    whenever the copy shares those dictionaries with the original.  The
    ``dataclasses.asdict`` snapshot must flag exactly that kind of edit, which is
    what ``diplomacy.benefit_sharing_matrix``, ``optimize`` and ``cli.json_safe``
    would do if they mutated the basin.
    """
    before = dataclasses.asdict(basin)
    assert dataclasses.asdict(basin) == before  # stable without any edit
    delta = basin.riparians[2].demand
    assert delta.value_usd_per_m3[Sector.MUNICIPAL] == 1.5
    delta.value_usd_per_m3[Sector.MUNICIPAL] = 0.0  # the mutation the guard must catch
    assert dataclasses.asdict(basin) != before
    delta.value_usd_per_m3[Sector.MUNICIPAL] = 1.5  # restore -> equal again
    assert dataclasses.asdict(basin) == before
    delta.consumption_fraction[Sector.AGRICULTURAL] = 0.99
    assert dataclasses.asdict(basin) != before
    delta.consumption_fraction[Sector.AGRICULTURAL] = 0.6
    assert dataclasses.asdict(basin) == before
    # the snapshot itself is independent of the basin's dictionaries
    before["riparians"][0]["demand"]["value_usd_per_m3"][Sector.MUNICIPAL] = -1.0
    assert basin.riparians[0].demand.value_usd_per_m3[Sector.MUNICIPAL] == 1.5


def test_treaty_entitlements_as_claims(basin):
    claims = {r.name: r.treaty_allocation_mm3 for r in basin}
    estate = 0.8 * sum(claims.values())  # a dry year
    t = talmud(estate, claims)
    ap = adjusted_proportional(estate, claims)
    assert sum(t.values()) == pytest.approx(estate)
    assert sum(ap.values()) == pytest.approx(estate)
    names = basin.names()
    v = bankruptcy_game(estate, claims)
    nuc = nucleolus(names, v)
    assert nuc == pytest.approx(t, rel=1e-7)
    assert is_in_core(nuc, names, v, tol=1e-6)
