"""Tests for :mod:`wefnexus.diplomacy` - hydro-politics indicators."""
import copy
import math

import numpy as np
import pytest

from wefnexus.allocation import RULES, apply_rule, gini, satisfaction
from wefnexus.data import example_basin
from wefnexus.energy import hydropower_gwh
from wefnexus.models import Basin, EnergySystem, Riparian, Sector, WaterDemand
from wefnexus.water import natural_flows, route_basin, sdg_642_water_stress
from wefnexus.diplomacy import (
    BAR_SCALE,
    CONFLICT_RISK_CATEGORIES,
    CONFLICT_RISK_WEIGHTS,
    HEGEMONY_PILLARS,
    TREATY_PRESENCE_CREDIT,
    TREATY_RESILIENCE_WEIGHTS,
    TWINS_CONFLICT_THRESHOLDS,
    TWINS_COOPERATION_THRESHOLDS,
    VARIABILITY_CV_REFERENCE,
    BarEvent,
    Treaty,
    bankruptcy_estate,
    bar_event_summary,
    bar_label,
    benefit_sharing_matrix,
    compare_allocation_rules,
    compliance_summary,
    conflict_intensity,
    conflict_risk_index,
    cooperation_index,
    cooperation_intensity,
    hydro_hegemony,
    negotiate,
    power_asymmetry,
    risk_category,
    treaty_compliance,
    treaty_from_basin,
    treaty_resilience,
    twins_classification,
    twins_conflict_class,
    twins_cooperation_class,
    water_dependency,
)

NAMES = ["Highland", "Midland", "Delta"]
ALLOC = {"Highland": 2_500.0, "Midland": 9_000.0, "Delta": 16_000.0}
ALL_FEATURES = dict(
    variable_allocation=True,
    drought_provisions=True,
    data_sharing=True,
    joint_institution=True,
    dispute_resolution=True,
    benefit_sharing=True,
    review_period_years=10,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_riparian(name, local=1000.0, mun=0.0, ag=0.0, env=0.0, material=0.5, bargaining=0.5,
                  ideational=0.5, treaty=None, capacity=0.0, storage=0.0, hydro_mw=0.0, head=0.0):
    return Riparian(
        name=name,
        population=1e6,
        gdp_usd=1e9,
        local_inflow_mm3=local,
        demand=WaterDemand(municipal=mun, agricultural=ag, environmental=env),
        energy=EnergySystem(hydropower_capacity_mw=hydro_mw, hydropower_head_m=head),
        reservoir_capacity_mm3=capacity,
        reservoir_storage_mm3=storage,
        material_power=material,
        bargaining_power=bargaining,
        ideational_power=ideational,
        treaty_allocation_mm3=treaty,
    )


def chain(n, **kw):
    """Basin with ``n`` identical riparians A, B, C, ..."""
    return Basin("chain", [make_riparian(chr(65 + i), **kw) for i in range(n)], headwater_inflow_mm3=500.0)


def bare_treaty(parties=NAMES, allocations=ALLOC, **features):
    return Treaty("bare", list(parties), dict(allocations), **features)


def full_treaty(parties=NAMES, allocations=ALLOC, **extra):
    kw = dict(ALL_FEATURES)
    kw.update(extra)
    return Treaty("full", list(parties), dict(allocations), **kw)


EVENTS = [
    BarEvent(2000, ["Highland", "Delta"], -3, issue="quantity"),
    BarEvent(2003, ["Midland", "Delta"], 4, issue="infrastructure"),
    BarEvent(2005, ["Highland", "Midland", "Delta"], 6, issue="treaty"),
    BarEvent(2010, ["Highland", "Delta"], -1, issue="quality"),
]


def assert_unchanged(basin, snapshot):
    assert basin == snapshot, "basin was mutated"


# ---------------------------------------------------------------------------
# BAR scale and events
# ---------------------------------------------------------------------------
def test_bar_scale_covers_minus7_to_plus7():
    assert sorted(BAR_SCALE) == list(range(-7, 8))
    assert all(isinstance(v, str) and v for v in BAR_SCALE.values())
    assert "war" in BAR_SCALE[-7].lower()
    assert "treaty" in BAR_SCALE[6].lower()
    assert "unification" in BAR_SCALE[7].lower()
    assert bar_label(0) == BAR_SCALE[0]
    assert bar_label(np.int64(-5)) == BAR_SCALE[-5]


@pytest.mark.parametrize("bad", [8, -8, 2.5, True, "3", None])
def test_bar_label_rejects_bad_scale(bad):
    with pytest.raises(ValueError):
        bar_label(bad)


def test_bar_event_validation_and_properties():
    e = BarEvent(2001, ("A", "B"), 3.0, issue="x")
    assert e.scale == 3 and isinstance(e.scale, int)
    assert e.parties == ["A", "B"]
    assert e.label == BAR_SCALE[3]
    assert e.is_cooperative and not e.is_conflictive
    assert e.involves(["B"]) and not e.involves(["C"])
    assert BarEvent(2001, [], -2).is_conflictive
    for kw in [
        dict(year=2000, parties=["A"], scale=8),
        dict(year=2000, parties=["A"], scale=-8),
        dict(year=2000, parties=["A"], scale=True),
        dict(year=2000, parties=["A"], scale=1.5),
        dict(year=2000.5, parties=["A"], scale=1),
        dict(year=2000, parties="A", scale=1),
        dict(year=2000, parties=[1], scale=1),
        dict(year=2000, parties=["A"], scale=1, issue=3),
    ]:
        with pytest.raises(ValueError):
            BarEvent(**kw)


# ---------------------------------------------------------------------------
# Treaty container
# ---------------------------------------------------------------------------
def test_treaty_normalises_and_validates():
    t = Treaty("t", ("A", "B"), {"A": 10}, drought_provisions=np.bool_(True), review_period_years=5.0,
               year_signed=1999.0, reference_flow_mm3=100)
    assert t.parties == ["A", "B"]
    assert t.allocations_mm3 == {"A": 10.0}
    assert t.drought_provisions is True and t.review_period_years == 5 and t.year_signed == 1999
    assert t.covers("A") and not t.covers("Z")
    assert t.features() == {
        "variable_allocation": False,
        "drought_provisions": True,
        "data_sharing": False,
        "joint_institution": False,
        "dispute_resolution": False,
        "benefit_sharing": False,
        "review_period": True,
    }
    d = t.to_dict()
    assert d["parties"] == ["A", "B"] and d["resilience"] == pytest.approx(treaty_resilience(t))


@pytest.mark.parametrize(
    "kw",
    [
        dict(name="t", parties=[], allocations_mm3={}),
        dict(name="t", parties=["A", "A"], allocations_mm3={}),
        dict(name="t", parties="A", allocations_mm3={}),
        dict(name="t", parties=["A", ""], allocations_mm3={}),
        dict(name="t", parties=["A"], allocations_mm3={"B": 1.0}),
        dict(name="t", parties=["A"], allocations_mm3={"A": -1.0}),
        dict(name="t", parties=["A"], allocations_mm3={"A": math.inf}),
        dict(name="t", parties=["A"], allocations_mm3=[("A", 1.0)]),
        dict(name="t", parties=["A"], allocations_mm3={}, data_sharing="yes"),
        dict(name="t", parties=["A"], allocations_mm3={}, review_period_years=0),
        dict(name="t", parties=["A"], allocations_mm3={}, review_period_years=2.5),
        dict(name="t", parties=["A"], allocations_mm3={}, reference_flow_mm3=0.0),
        dict(name=5, parties=["A"], allocations_mm3={}),
    ],
)
def test_treaty_rejects_bad_fields(kw):
    with pytest.raises(ValueError):
        Treaty(**kw)


def test_treaty_entitlements_fixed_and_variable():
    fixed = Treaty("f", ["A", "B"], {"A": 100.0}, variable_allocation=False, reference_flow_mm3=1000.0)
    assert fixed.entitlements(500.0) == {"A": 100.0}
    var = Treaty("v", ["A", "B"], {"A": 100.0}, variable_allocation=True, reference_flow_mm3=1000.0)
    assert var.entitlements(500.0) == {"A": 50.0}
    assert var.entitlements(2000.0) == {"A": 200.0}
    assert var.entitlements() == {"A": 100.0}  # no flow given -> fixed volumes
    no_ref = Treaty("n", ["A"], {"A": 100.0}, variable_allocation=True)
    assert no_ref.entitlements(500.0) == {"A": 100.0}
    with pytest.raises(ValueError):
        var.entitlements(-1.0)
    # the returned dict is a copy
    var.entitlements()["A"] = 0.0
    assert var.allocations_mm3["A"] == 100.0


def test_treaty_edits_are_revalidated_without_mutation():
    t = bare_treaty()
    t.parties = ("Highland", "Midland", "Delta")
    treaty_resilience(t)
    assert isinstance(t.parties, tuple)  # checks do not rewrite the user's object
    t.allocations_mm3["Highland"] = -5.0
    with pytest.raises(ValueError):
        treaty_resilience(t)


def test_treaty_from_basin(basin):
    t = treaty_from_basin(basin)
    assert t.parties == NAMES and t.allocations_mm3 == ALLOC
    assert treaty_resilience(t) == 0.0
    t2 = treaty_from_basin(basin, name="Azura Accord", drought_provisions=True)
    assert t2.name == "Azura Accord" and t2.drought_provisions
    b = basin.copy()
    b.riparian("Midland").treaty_allocation_mm3 = None
    assert treaty_from_basin(b).parties == ["Highland", "Delta"]
    for r in b:
        r.treaty_allocation_mm3 = None
    assert treaty_from_basin(b) is None
    with pytest.raises(ValueError):
        treaty_from_basin("basin")


# ---------------------------------------------------------------------------
# hydro-hegemony
# ---------------------------------------------------------------------------
def test_geographic_pillar_upstream_one_downstream_zero(basin):
    h = {r.name: hydro_hegemony(r, basin) for r in basin}
    assert h["Highland"]["geographic"] == 1.0
    assert h["Midland"]["geographic"] == pytest.approx(0.5)
    assert h["Delta"]["geographic"] == 0.0
    for name, r in zip(NAMES, basin):
        assert set(h[name]) == set(HEGEMONY_PILLARS) | {"score"}
        assert h[name]["material"] == r.material_power
        assert h[name]["bargaining"] == r.bargaining_power
        assert h[name]["ideational"] == r.ideational_power
        expected = (h[name]["geographic"] + r.material_power + r.bargaining_power + r.ideational_power) / 4
        assert h[name]["score"] == pytest.approx(expected)
        assert all(0.0 <= v <= 1.0 for v in h[name].values())


def test_geographic_pillar_decreases_with_position():
    b = chain(5)
    geo = [hydro_hegemony(r, b)["geographic"] for r in b]
    assert geo == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    single = chain(1)
    assert hydro_hegemony(single.riparians[0], single)["geographic"] == 1.0


def test_hydro_hegemony_weights_and_whatif(basin):
    delta = basin.riparian("Delta")
    only_geo = hydro_hegemony(delta, basin, weights={"material": 0, "bargaining": 0, "ideational": 0})
    assert only_geo["score"] == 0.0
    heavy = hydro_hegemony(delta, basin, weights={"material": 3})
    expected = (0.0 + 3 * 0.8 + 0.5 + 0.7) / 6
    assert heavy["score"] == pytest.approx(expected)
    # a modified copy keeps the basin position but uses its own power values
    stronger = delta.copy(material_power=1.0)
    out = hydro_hegemony(stronger, basin)
    assert out["geographic"] == 0.0 and out["material"] == 1.0
    assert basin.riparian("Delta").material_power == 0.8


def test_hydro_hegemony_validation(basin):
    delta = basin.riparian("Delta")
    stranger = make_riparian("Nowhere")
    with pytest.raises(ValueError):
        hydro_hegemony(stranger, basin)
    with pytest.raises(ValueError):
        hydro_hegemony("Delta", basin)
    with pytest.raises(ValueError):
        hydro_hegemony(delta, "basin")
    with pytest.raises(ValueError):
        hydro_hegemony(delta, basin, weights={"geo": 1})
    with pytest.raises(ValueError):
        hydro_hegemony(delta, basin, weights={"geographic": -1})
    with pytest.raises(ValueError):
        hydro_hegemony(delta, basin, weights={k: 0 for k in HEGEMONY_PILLARS})
    with pytest.raises(ValueError):
        hydro_hegemony(delta.copy(material_power=1.5), basin)
    with pytest.raises(ValueError):
        hydro_hegemony(delta.copy(bargaining_power=-0.1), basin)


def test_power_asymmetry_bounds_and_closed_form(basin):
    a = power_asymmetry(basin)
    scores = [hydro_hegemony(r, basin)["score"] for r in basin]
    assert 0.0 <= a <= 1.0
    assert a == pytest.approx(max(scores) - min(scores))
    assert a == pytest.approx(0.6125 - 0.5)
    assert power_asymmetry(chain(1)) == 0.0
    # identical riparians differ only by position
    assert power_asymmetry(chain(3)) == pytest.approx(0.25)
    assert power_asymmetry(chain(3), weights={"geographic": 0}) == 0.0
    extreme = Basin("x", [make_riparian("up", material=1, bargaining=1, ideational=1),
                          make_riparian("down", material=0, bargaining=0, ideational=0)])
    assert power_asymmetry(extreme) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        power_asymmetry(None)


# ---------------------------------------------------------------------------
# cooperation / conflict indices
# ---------------------------------------------------------------------------
def test_cooperation_and_conflict_closed_form():
    assert cooperation_index(EVENTS) == pytest.approx((-3 + 4 + 6 - 1) / 4)
    assert conflict_intensity(EVENTS) == pytest.approx((3 + 1) / 2)
    # window of 5 years ending at the latest event (2010): only 2010 is kept
    assert cooperation_index(EVENTS, window_years=5) == pytest.approx(-1.0)
    assert conflict_intensity(EVENTS, window_years=5) == pytest.approx(1.0)
    # explicit `until` drops later events
    assert cooperation_index(EVENTS, until=2005) == pytest.approx((-3 + 4 + 6) / 3)
    assert cooperation_index(EVENTS, window_years=3, until=2005) == pytest.approx((4 + 6) / 2)
    assert conflict_intensity(EVENTS, until=2004) == pytest.approx(3.0)
    # window before any event
    assert cooperation_index(EVENTS, until=1990) == 0.0
    assert conflict_intensity(EVENTS, until=1990) == 0.0


def test_indices_edge_cases():
    assert cooperation_index([]) == 0.0
    assert conflict_intensity([]) == 0.0
    assert cooperation_index(()) == 0.0
    coop_only = [BarEvent(2000, [], 2), BarEvent(2001, [], 5)]
    assert conflict_intensity(coop_only) == 0.0
    assert cooperation_index(coop_only) == pytest.approx(3.5)
    war = [BarEvent(2000, [], -7)]
    assert cooperation_index(war) == -7.0 and conflict_intensity(war) == 7.0
    assert -7.0 <= cooperation_index(EVENTS) <= 7.0
    assert 0.0 <= conflict_intensity(EVENTS) <= 7.0
    # generators are accepted
    assert cooperation_index(e for e in EVENTS) == pytest.approx(1.5)


def test_indices_party_filter():
    assert cooperation_index(EVENTS, parties=["Midland"]) == pytest.approx((4 + 6) / 2)
    assert conflict_intensity(EVENTS, parties=["Midland"]) == 0.0
    assert conflict_intensity(EVENTS, parties=["Highland"]) == pytest.approx(2.0)
    assert cooperation_index(EVENTS, parties=["Nobody"]) == 0.0
    with pytest.raises(ValueError):
        cooperation_index(EVENTS, parties="Midland")


def test_indices_validation():
    with pytest.raises(ValueError):
        cooperation_index(EVENTS[0])
    with pytest.raises(ValueError):
        cooperation_index([EVENTS[0], "event"])
    with pytest.raises(ValueError):
        cooperation_index(EVENTS, window_years=0)
    with pytest.raises(ValueError):
        cooperation_index(EVENTS, window_years=-2)
    with pytest.raises(ValueError):
        cooperation_index(EVENTS, window_years=True)
    with pytest.raises(ValueError):
        cooperation_index(EVENTS, until=2000.5)
    with pytest.raises(ValueError):
        conflict_intensity(5)
    edited = BarEvent(2000, [], 1)
    edited.scale = 9
    with pytest.raises(ValueError):
        conflict_intensity([edited])


def test_bar_event_summary():
    s = bar_event_summary(EVENTS)
    assert s["n_events"] == 4 and s["n_cooperative"] == 2 and s["n_conflictive"] == 2 and s["n_neutral"] == 0
    assert s["share_conflictive"] == pytest.approx(0.5)
    assert s["cooperation_index"] == pytest.approx(1.5)        # net mean of all four events
    assert s["cooperation_intensity"] == pytest.approx(5.0)    # mean of the two cooperative events (+4, +6)
    assert s["conflict_intensity"] == pytest.approx(2.0)       # mean |scale| of the two conflictive events
    assert s["most_conflictive"] == -3 and s["most_cooperative"] == 6
    # the TWINS cell is scored from the two intensities, never from the net index
    assert s["twins"] == twins_classification(5.0, 2.0) == "risk_averting/politicised"
    empty = bar_event_summary([])
    assert empty["n_events"] == 0 and empty["most_conflictive"] is None
    assert empty["cooperation_intensity"] == 0.0 and empty["conflict_intensity"] == 0.0
    assert empty["share_conflictive"] == 0.0 and empty["twins"] == "none/none"
    neutral = bar_event_summary([BarEvent(2000, [], 0)])
    assert neutral["n_neutral"] == 1 and neutral["twins"] == "none/none"


def test_cooperation_intensity_closed_form_windows_and_parties():
    assert cooperation_intensity(EVENTS) == pytest.approx((4 + 6) / 2)
    assert cooperation_intensity(EVENTS, window_years=5) == 0.0  # only the 2010 (-1) event is in the window
    assert cooperation_intensity(EVENTS, until=2004) == pytest.approx(4.0)
    assert cooperation_intensity(EVENTS, window_years=3, until=2005) == pytest.approx(5.0)
    assert cooperation_intensity(EVENTS, until=1990) == 0.0
    assert cooperation_intensity(EVENTS, parties=["Midland"]) == pytest.approx(5.0)
    assert cooperation_intensity(EVENTS, parties=["Nobody"]) == 0.0
    assert cooperation_intensity([]) == 0.0 and cooperation_intensity(()) == 0.0
    assert cooperation_intensity([BarEvent(2000, [], -7), BarEvent(2001, [], 0)]) == 0.0
    assert cooperation_intensity(e for e in EVENTS) == pytest.approx(5.0)
    for record in ([BarEvent(2000, [], 7)], EVENTS, [BarEvent(2000, [], 1)], []):
        assert 0.0 <= cooperation_intensity(record) <= 7.0
    with pytest.raises(ValueError):
        cooperation_intensity(EVENTS[0])
    with pytest.raises(ValueError):
        cooperation_intensity([EVENTS[0], "event"])
    with pytest.raises(ValueError):
        cooperation_intensity(EVENTS, window_years=0)
    with pytest.raises(ValueError):
        cooperation_intensity(EVENTS, parties="Midland")
    edited = BarEvent(2000, [], 1)
    edited.scale = 9
    with pytest.raises(ValueError):
        cooperation_intensity([edited])


def test_conflict_events_never_lower_cooperation_intensity():
    """The two TWINS axes are independent: adding a hostile event changes only the conflict axis."""
    coop = [BarEvent(2000, [], 3), BarEvent(2001, [], 5)]
    assert cooperation_intensity(coop) == pytest.approx(4.0)
    for scale in range(-7, 1):
        with_hostile = coop + [BarEvent(2002, [], scale)]
        assert cooperation_intensity(with_hostile) == pytest.approx(4.0)
        assert cooperation_index(with_hostile) <= cooperation_index(coop)  # the net index does move
    conf = [BarEvent(2000, [], -4)]
    for scale in range(0, 8):
        assert conflict_intensity(conf + [BarEvent(2001, [], scale)]) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# TWINS matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cooperation, conflict, expected",
    [
        (0, 0, "none/none"),
        (7, 0, "risk_taking/none"),
        (0, 7, "none/violent"),
        (7, 7, "risk_taking/violent"),
        (1, 1, "confrontation/non-politicised"),
        (1.99, 1.99, "confrontation/non-politicised"),
        (2, 2, "ad_hoc/politicised"),
        (2.99, 3.99, "ad_hoc/politicised"),
        (3, 4, "technical/securitised"),
        (3.99, 4.99, "technical/securitised"),
        (4, 5, "risk_averting/violent"),
        (5.99, 0.5, "risk_averting/non-politicised"),
        (6, 5, "risk_taking/violent"),
        (np.float64(6.5), np.int64(3), "risk_taking/politicised"),
    ],
)
def test_twins_matrix_cells(cooperation, conflict, expected):
    assert twins_classification(cooperation, conflict) == expected


def test_twins_classes_are_monotone_and_exhaustive():
    coop_order = ["none", "confrontation", "ad_hoc", "technical", "risk_averting", "risk_taking"]
    conf_order = ["none", "non-politicised", "politicised", "securitised", "violent"]
    last = -1
    for c in np.linspace(0, 7, 71):
        idx = coop_order.index(twins_cooperation_class(float(c)))
        assert idx >= last
        last = idx
    last = -1
    for k in np.linspace(0, 7, 71):
        idx = conf_order.index(twins_conflict_class(float(k)))
        assert idx >= last
        last = idx
    assert set(TWINS_COOPERATION_THRESHOLDS) == set(coop_order[1:])
    assert set(TWINS_CONFLICT_THRESHOLDS) == set(conf_order[1:])
    assert list(TWINS_COOPERATION_THRESHOLDS.values()) == sorted(TWINS_COOPERATION_THRESHOLDS.values())
    assert twins_cooperation_class(0.0) == "none"
    assert twins_cooperation_class(1e-9) == "confrontation"
    assert twins_conflict_class(0.0) == "none"
    assert twins_conflict_class(1e-9) == "non-politicised"


@pytest.mark.parametrize(
    "cooperation, conflict",
    [(8, 0), (-8, 0), (-1, 0), (-0.5, 0), (0, -1), (0, 8), (math.nan, 0), (0, math.inf), ("a", 0), (None, 0)],
)
def test_twins_rejects_out_of_range(cooperation, conflict):
    with pytest.raises(ValueError):
        twins_classification(cooperation, conflict)


def test_twins_axes_are_independent_review_example():
    """A treaty (+6) and extensive war acts (-6) in one window: the net mean is 0, TWINS keeps both."""
    events = [BarEvent(2000, [], 6), BarEvent(2001, [], -6)]
    assert cooperation_index(events) == 0.0
    assert cooperation_intensity(events) == pytest.approx(6.0)
    assert conflict_intensity(events) == pytest.approx(6.0)
    s = bar_event_summary(events)
    assert s["twins"] == "risk_taking/violent"
    assert s["cooperation_index"] == 0.0 and s["cooperation_intensity"] == pytest.approx(6.0)
    # the net index is not a TWINS coordinate: a negative one is rejected loudly
    with pytest.raises(ValueError):
        twins_classification(cooperation_index([BarEvent(2000, [], -2)]), 0.0)


@pytest.mark.parametrize(
    "scale, expected",
    [(1, "confrontation"), (2, "ad_hoc"), (3, "technical"), (4, "risk_averting"), (5, "risk_averting"),
     (6, "risk_taking"), (7, "risk_taking")],
)
def test_each_positive_bar_level_reaches_its_twins_class(scale, expected):
    events = [BarEvent(2000, [], scale), BarEvent(2001, [], -7)]  # a war alongside must not matter
    assert twins_cooperation_class(cooperation_intensity(events)) == expected
    assert bar_event_summary(events)["twins"] == f"{expected}/violent"
    assert bar_event_summary(events[:1])["twins"] == f"{expected}/none"


def test_no_cooperative_events_is_none_not_confrontation():
    assert twins_cooperation_class(0.0) == "none"
    assert twins_classification(0, 0) == "none/none"
    assert bar_event_summary([BarEvent(2000, [], -3)])["twins"] == "none/politicised"
    assert bar_event_summary([BarEvent(2000, [], 0)])["twins"] == "none/none"
    assert bar_event_summary([BarEvent(2000, [], -7), BarEvent(2001, [], -1)])["twins"] == "none/securitised"


# ---------------------------------------------------------------------------
# treaty resilience
# ---------------------------------------------------------------------------
def test_treaty_resilience_extremes():
    assert treaty_resilience(bare_treaty()) == 0.0
    assert treaty_resilience(full_treaty()) == 1.0
    assert sum(TREATY_RESILIENCE_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("feature", list(TREATY_RESILIENCE_WEIGHTS))
def test_treaty_resilience_single_feature_equals_its_weight(feature):
    kw = {"review_period_years": 5} if feature == "review_period" else {feature: True}
    t = bare_treaty(**kw)
    assert treaty_resilience(t) == pytest.approx(TREATY_RESILIENCE_WEIGHTS[feature])


def test_treaty_resilience_monotone_and_weights():
    features = list(ALL_FEATURES)
    previous = 0.0
    kw = {}
    for f in features:
        kw[f] = ALL_FEATURES[f]
        score = treaty_resilience(Treaty("t", ["A"], {}, **kw))
        assert 0.0 <= score <= 1.0
        assert score > previous
        previous = score
    assert previous == pytest.approx(1.0)
    # weights override: only data sharing counts
    t = bare_treaty(data_sharing=True, drought_provisions=True)
    w = {k: 0.0 for k in TREATY_RESILIENCE_WEIGHTS}
    w["data_sharing"] = 1.0
    assert treaty_resilience(t, weights=w) == 1.0
    w["drought_provisions"] = 1.0
    w["joint_institution"] = 2.0
    assert treaty_resilience(t, weights=w) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        treaty_resilience(t, weights={"unknown": 1.0})
    with pytest.raises(ValueError):
        treaty_resilience(t, weights={"data_sharing": -1.0})
    with pytest.raises(ValueError):
        treaty_resilience(t, weights={k: 0.0 for k in TREATY_RESILIENCE_WEIGHTS})
    with pytest.raises(ValueError):
        treaty_resilience({"name": "dict"})


# ---------------------------------------------------------------------------
# treaty compliance
# ---------------------------------------------------------------------------
def test_treaty_compliance_example_basin(basin):
    bal = route_basin(basin)
    out = treaty_compliance(bare_treaty(), bal)
    assert list(out) == NAMES
    for name in NAMES:
        row = out[name]
        reach = bal.reach(name)
        assert row["entitlement"] == ALLOC[name]
        assert row["withdrawal"] == pytest.approx(reach.total_withdrawal())
        assert row["compliance_ratio"] == pytest.approx(reach.total_withdrawal() / ALLOC[name])
        assert row["compliance_ratio"] <= 1.0 + 1e-9  # routing honoured the caps
        assert row["over_abstraction_mm3"] == 0.0 and row["compliant"] == 1.0
        assert row["inflow"] == pytest.approx(reach.inflow_mm3)
        assert row["upstream_inflow"] == pytest.approx(reach.upstream_inflow_mm3)
        assert row["delivery_ratio"] == pytest.approx(reach.inflow_mm3 / ALLOC[name])
    assert out["Delta"]["compliance_ratio"] == pytest.approx(1.0)  # cap binds for Delta
    assert all(out[n]["delivered"] == 1.0 for n in NAMES)
    summary = compliance_summary(out)
    assert summary["share_compliant"] == 1.0 and summary["share_delivered"] == 1.0
    assert summary["operational"] == 1.0 and summary["n_parties"] == 3.0


def test_treaty_compliance_detects_over_abstraction_and_shortfall(basin):
    # Highland withdraws without a cap against a 1000 Mm3 entitlement
    bal = route_basin(basin, entitlements={"Highland": None})
    t = bare_treaty(allocations={"Highland": 1_000.0, "Midland": 9_000.0, "Delta": 16_000.0})
    out = treaty_compliance(t, bal)
    h = out["Highland"]
    assert h["withdrawal"] == pytest.approx(1_500.0)
    assert h["compliance_ratio"] == pytest.approx(1.5)
    assert h["over_abstraction_mm3"] == pytest.approx(500.0)
    assert h["compliant"] == 0.0
    # severe drought: Delta no longer receives its entitlement as inflow
    dry = treaty_compliance(bare_treaty(), route_basin(basin, 0.6))
    d = dry["Delta"]
    assert d["delivery_ratio"] < 1.0 and d["delivered"] == 0.0
    assert d["delivery_shortfall_mm3"] == pytest.approx(16_000.0 - d["inflow"])
    s = compliance_summary(dry)
    assert s["share_delivered"] == pytest.approx(2 / 3) and s["operational"] == 0.0
    assert s["total_delivery_shortfall_mm3"] == pytest.approx(d["delivery_shortfall_mm3"])


def test_treaty_compliance_conventions(basin):
    bal = route_basin(basin)
    # party without a volumetric allocation: unlimited, nothing to deliver
    t = Treaty("partial", NAMES, {"Delta": 16_000.0})
    out = treaty_compliance(t, bal)
    assert out["Highland"]["entitlement"] == math.inf
    assert out["Highland"]["compliance_ratio"] == 0.0 and out["Highland"]["compliant"] == 1.0
    assert out["Highland"]["delivery_ratio"] == 1.0 and out["Highland"]["delivered"] == 1.0
    assert out["Delta"]["entitlement"] == 16_000.0
    # zero entitlement: compliant only if nothing withdrawn
    zero = Treaty("zero", NAMES, {"Highland": 0.0})
    out = treaty_compliance(zero, bal)
    assert out["Highland"]["compliance_ratio"] == math.inf and out["Highland"]["compliant"] == 0.0
    assert out["Highland"]["over_abstraction_mm3"] == pytest.approx(1_500.0)
    quiet = route_basin(basin, entitlements={"Highland": 0.0})
    out = treaty_compliance(zero, quiet)
    assert out["Highland"]["compliance_ratio"] == 1.0 and out["Highland"]["compliant"] == 1.0
    # variable allocation scales with the routed natural flow
    var = Treaty("var", NAMES, ALLOC, variable_allocation=True, reference_flow_mm3=28_000.0)
    dry = route_basin(basin, 0.5)
    out = treaty_compliance(var, dry)
    assert out["Delta"]["entitlement"] == pytest.approx(8_000.0)
    assert out["Delta"]["compliance_ratio"] == pytest.approx(dry.reach("Delta").total_withdrawal() / 8_000.0)


def test_treaty_compliance_validation(basin):
    bal = route_basin(basin)
    with pytest.raises(ValueError):
        treaty_compliance(Treaty("x", ["Highland", "Elsewhere"], {}), bal)
    with pytest.raises(ValueError):
        treaty_compliance(bare_treaty(), "balance")
    with pytest.raises(ValueError):
        treaty_compliance("treaty", bal)
    with pytest.raises(ValueError):
        treaty_compliance(bare_treaty(), bal, tol=-1)
    with pytest.raises(ValueError):
        compliance_summary({"Highland": {"compliant": 1.0}})
    with pytest.raises(ValueError):
        compliance_summary([1, 2])
    assert compliance_summary({})["operational"] == 1.0


# ---------------------------------------------------------------------------
# dependency
# ---------------------------------------------------------------------------
def test_water_dependency_natural_closed_form(basin):
    dep = water_dependency(basin)
    assert list(dep) == NAMES
    assert dep["Highland"] == pytest.approx(2_000 / 20_000)
    assert dep["Midland"] == pytest.approx(20_000 / 27_000)
    assert dep["Delta"] == pytest.approx(27_000 / 28_000)
    assert all(0.0 <= v <= 1.0 for v in dep.values())


def test_water_dependency_from_balance(basin):
    bal = route_basin(basin, 0.8)
    dep = water_dependency(basin, bal)
    for reach in bal.reaches:
        assert dep[reach.name] == pytest.approx(reach.upstream_inflow_mm3 / reach.inflow_mm3)
    # consumption upstream lowers the downstream share compared with natural flow
    assert dep["Delta"] < water_dependency(basin)["Delta"]


def test_water_dependency_edge_cases():
    b = Basin("nohead", [make_riparian("A", local=100.0), make_riparian("B", local=0.0)])
    dep = water_dependency(b)
    assert dep == {"A": 0.0, "B": 1.0}
    dry = Basin("dry", [make_riparian("A", local=0.0), make_riparian("B", local=0.0)])
    assert water_dependency(dry) == {"A": 0.0, "B": 0.0}
    assert water_dependency(dry, route_basin(dry)) == {"A": 0.0, "B": 0.0}
    assert water_dependency(Basin("empty", [])) == {}
    other = Basin("other", [make_riparian("X")])
    with pytest.raises(ValueError):
        water_dependency(b, route_basin(other))
    with pytest.raises(ValueError):
        water_dependency(b, "balance")


# ---------------------------------------------------------------------------
# conflict risk index
# ---------------------------------------------------------------------------
def test_conflict_risk_index_example_basin(basin):
    bal = route_basin(basin)
    out = conflict_risk_index(basin, bal)
    assert 0.0 < out["score"] < 1.0
    assert out["category"] in ("low", "moderate", "high", "very_high")
    assert out["category"] == risk_category(out["score"])
    comps = out["components"]
    assert set(comps) == set(CONFLICT_RISK_WEIGHTS)
    assert all(0.0 <= v <= 1.0 for v in comps.values())
    w = out["weights"]
    expected = sum(w[k] * comps[k] for k in comps) / sum(w.values())
    assert out["score"] == pytest.approx(expected)
    # component closed forms
    assert comps["institutional"] == 1.0  # no treaty
    assert comps["conflict_history"] == 0.0  # no events
    assert comps["power_asymmetry"] == pytest.approx(power_asymmetry(basin))
    assert comps["variability"] == pytest.approx(min(basin.climate_cv / VARIABILITY_CV_REFERENCE, 1.0))
    assert comps["dependency"] == pytest.approx(np.mean(list(water_dependency(basin, bal).values())))
    assert comps["environmental"] == 0.0  # all env flows met at mean flow
    upstream_refill = sum(r.storage_refill_mm3 for r in bal.reaches[:-1])
    assert comps["dam_filling"] == pytest.approx(upstream_refill / bal.natural_flow_mm3)
    details = out["details"]
    stress = details["water_stress_pct"]
    delta = bal.reach("Delta")
    rip = basin.riparian("Delta")
    gw_share = rip.groundwater_abstraction_mm3 / (rip.groundwater_abstraction_mm3 + rip.energy.desalination_capacity_mm3)
    expected_stress = sdg_642_water_stress(
        delta.total_withdrawal() + delta.non_river_supply_mm3 * gw_share,
        delta.inflow_mm3 + rip.groundwater_recharge_mm3,
        delta.environmental_flow_mm3,
    )
    assert stress["Delta"] == pytest.approx(expected_stress)
    assert comps["water_stress"] == pytest.approx(np.mean([min(s / 100.0, 1.0) for s in stress.values()]))
    assert details["treaty_resilience"] is None and details["n_events"] == 0


def test_conflict_risk_rises_in_drought_and_falls_with_resilient_treaty(basin):
    wet = conflict_risk_index(basin, route_basin(basin, 1.0))["score"]
    dry = conflict_risk_index(basin, route_basin(basin, 0.6))["score"]
    assert dry > wet
    with_full = conflict_risk_index(basin, route_basin(basin, 1.0), treaty=full_treaty())["score"]
    with_bare = conflict_risk_index(basin, route_basin(basin, 1.0), treaty=bare_treaty())["score"]
    assert with_full < with_bare < wet
    assert with_bare == pytest.approx(wet - CONFLICT_RISK_WEIGHTS["institutional"] * TREATY_PRESENCE_CREDIT)
    assert with_full == pytest.approx(wet - CONFLICT_RISK_WEIGHTS["institutional"])
    # a treaty covering only one of three riparians removes a third of the credit
    partial = Treaty("p", ["Delta"], {"Delta": 16_000.0}, **ALL_FEATURES)
    with_partial = conflict_risk_index(basin, route_basin(basin), treaty=partial)
    assert with_partial["components"]["institutional"] == pytest.approx(1 - 1 / 3)
    assert with_partial["details"]["treaty_coverage"] == pytest.approx(1 / 3)


def test_conflict_risk_events_variability_and_weights(basin):
    bal = route_basin(basin)
    base = conflict_risk_index(basin, bal)["score"]
    hostile = conflict_risk_index(basin, bal, events=[BarEvent(2000, NAMES, -7)])
    assert hostile["components"]["conflict_history"] == pytest.approx(1.0)
    assert hostile["score"] == pytest.approx(base + CONFLICT_RISK_WEIGHTS["conflict_history"])
    friendly = conflict_risk_index(basin, bal, events=[BarEvent(2000, NAMES, 6)])
    assert friendly["components"]["conflict_history"] == 0.0 and friendly["score"] == pytest.approx(base)
    mixed = conflict_risk_index(basin, bal, events=EVENTS)
    assert mixed["components"]["conflict_history"] == pytest.approx((3 + 1) / (7 * 4))
    assert mixed["details"]["n_events"] == 4 and mixed["details"]["n_conflictive_events"] == 2
    recent = conflict_risk_index(basin, bal, events=EVENTS, window_years=5)
    assert recent["components"]["conflict_history"] == pytest.approx(1 / 7)
    # variability
    calm = conflict_risk_index(basin, bal, climate_cv=0.0)
    wild = conflict_risk_index(basin, bal, climate_cv=5.0)
    assert calm["components"]["variability"] == 0.0 and wild["components"]["variability"] == 1.0
    assert calm["score"] < base < wild["score"]
    # weights
    only_treaty = conflict_risk_index(basin, bal, weights={k: 0.0 for k in CONFLICT_RISK_WEIGHTS if k != "institutional"})
    assert only_treaty["score"] == pytest.approx(1.0) and only_treaty["category"] == "very_high"
    only_treaty_full = conflict_risk_index(
        basin, bal, treaty=full_treaty(),
        weights={k: 0.0 for k in CONFLICT_RISK_WEIGHTS if k != "institutional"},
    )
    assert only_treaty_full["score"] == 0.0 and only_treaty_full["category"] == "low"


def test_conflict_risk_environmental_and_dam_components():
    # a downstream riparian whose env flow cannot be met -> environmental risk
    up = make_riparian("Up", local=100.0, ag=100.0, env=0.0)
    down = make_riparian("Down", local=0.0, env=50.0)
    b = Basin("env", [up, down], headwater_inflow_mm3=0.0)
    bal = route_basin(b)
    out = conflict_risk_index(b, bal)
    reach = bal.reach("Down")
    assert not reach.env_flow_met
    assert out["components"]["environmental"] == pytest.approx(0.5 * max(50.0 - reach.outflow_mm3, 0.0) / 50.0)
    assert out["details"]["env_shortfall"]["Up"] == 0.0
    # dam filling: only reservoirs with someone downstream count
    dam_up = make_riparian("A", local=1000.0, capacity=1000.0, storage=0.0)
    dam_down = make_riparian("B", local=0.0, capacity=1000.0, storage=0.0)
    b2 = Basin("dam", [dam_up, dam_down], headwater_inflow_mm3=0.0)
    bal2 = route_basin(b2)
    out2 = conflict_risk_index(b2, bal2)
    assert bal2.reach("B").storage_refill_mm3 > 0.0
    assert out2["components"]["dam_filling"] == pytest.approx(bal2.reach("A").storage_refill_mm3 / 1000.0)
    assert out2["components"]["dam_filling"] == pytest.approx(0.25)
    single = Basin("one", [make_riparian("A", local=1000.0, capacity=1000.0)])
    assert conflict_risk_index(single, route_basin(single))["components"]["dam_filling"] == 0.0
    # zero flow and no storage: water stress saturates, dam filling 0
    dry = route_basin(b2, 0.0)
    out3 = conflict_risk_index(b2, dry)
    assert out3["components"]["dam_filling"] == 0.0
    assert out3["components"]["water_stress"] == 1.0
    assert 0.0 <= out3["score"] <= 1.0
    # empty basin
    empty = Basin("empty", [])
    out4 = conflict_risk_index(empty, route_basin(empty))
    assert 0.0 <= out4["score"] <= 1.0 and out4["components"]["dependency"] == 0.0


def test_conflict_risk_validation(basin):
    bal = route_basin(basin)
    other = chain(2)
    with pytest.raises(ValueError):
        conflict_risk_index(basin, route_basin(other))
    with pytest.raises(ValueError):
        conflict_risk_index(basin, bal, treaty="treaty")
    with pytest.raises(ValueError):
        conflict_risk_index(basin, bal, events=[1, 2])
    with pytest.raises(ValueError):
        conflict_risk_index(basin, bal, climate_cv=-0.1)
    with pytest.raises(ValueError):
        conflict_risk_index(basin, bal, weights={"stress": 1.0})
    with pytest.raises(ValueError):
        conflict_risk_index(basin, bal, weights={k: 0.0 for k in CONFLICT_RISK_WEIGHTS})
    with pytest.raises(ValueError):
        conflict_risk_index("basin", bal)


def test_risk_category_thresholds():
    assert risk_category(0.0) == "low"
    assert risk_category(CONFLICT_RISK_CATEGORIES["low"] - 1e-9) == "low"
    assert risk_category(CONFLICT_RISK_CATEGORIES["low"]) == "moderate"
    assert risk_category(CONFLICT_RISK_CATEGORIES["moderate"]) == "high"
    assert risk_category(CONFLICT_RISK_CATEGORIES["high"]) == "very_high"
    assert risk_category(1.0) == "very_high"
    for bad in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError):
            risk_category(bad)


# ---------------------------------------------------------------------------
# benefit sharing
# ---------------------------------------------------------------------------
def test_benefit_sharing_matrix_example(basin):
    unilateral = route_basin(basin, entitlements={n: None for n in NAMES})
    coop = route_basin(basin, entitlements=negotiate(basin)["proposal"])
    out = benefit_sharing_matrix(basin, unilateral, coop)
    risk = conflict_risk_index(basin, unilateral)["score"]
    assert list(out) == NAMES
    for name, rip in zip(NAMES, basin):
        row = out[name]
        reach = unilateral.reach(name)
        assert set(row) >= {"to_the_river", "from_the_river", "because_of_the_river", "beyond_the_river"}
        assert 0.0 <= row["to_the_river"] <= 1.0
        assert row["to_the_river"] == pytest.approx(min(reach.outflow_mm3 / reach.environmental_flow_mm3, 1.0))
        water = sum(reach.withdrawals[s] * rip.demand.value_usd_per_m3[s] * 1e6 for s in reach.withdrawals)
        assert row["water_value_usd"] == pytest.approx(water)
        es = rip.energy
        gwh = hydropower_gwh(reach.outflow_mm3 * es.turbined_fraction, es.hydropower_head_m,
                             es.turbine_efficiency, es.hydropower_capacity_mw)
        assert row["hydropower_gwh"] == pytest.approx(gwh)
        assert row["hydropower_value_usd"] == pytest.approx(gwh * 1e6 * 0.05)
        assert row["from_the_river"] == pytest.approx(water + gwh * 1e6 * 0.05)
        assert row["because_of_the_river"] == pytest.approx(1.0 - risk)
        coop_row = benefit_sharing_matrix(basin, coop)[name]
        assert row["beyond_the_river"] == pytest.approx(coop_row["from_the_river"] - row["from_the_river"])
    # without a cooperative balance the catalytic benefit is zero
    assert all(r["beyond_the_river"] == 0.0 for r in benefit_sharing_matrix(basin, unilateral).values())
    # a different hydropower price scales only the hydropower value
    cheap = benefit_sharing_matrix(basin, unilateral, hydropower_value_usd_per_kwh=0.0)
    assert cheap["Highland"]["from_the_river"] == pytest.approx(out["Highland"]["water_value_usd"])


def test_benefit_sharing_edge_cases():
    b = Basin("plain", [make_riparian("A", local=100.0, mun=10.0, env=0.0),
                        make_riparian("B", local=0.0, env=1_000.0)])
    bal = route_basin(b)
    out = benefit_sharing_matrix(b, bal)
    assert out["A"]["to_the_river"] == 1.0  # no requirement
    assert out["B"]["to_the_river"] == pytest.approx(bal.reach("B").outflow_mm3 / 1_000.0)
    assert out["B"]["to_the_river"] < 1.0
    assert out["A"]["hydropower_gwh"] == 0.0  # no plant
    assert out["A"]["from_the_river"] == pytest.approx(10.0 * 1.5 * 1e6)
    with pytest.raises(ValueError):
        benefit_sharing_matrix(b, bal, hydropower_value_usd_per_kwh=-1.0)
    with pytest.raises(ValueError):
        benefit_sharing_matrix(b, bal, route_basin(chain(3)))  # different riparians
    with pytest.raises(ValueError):
        benefit_sharing_matrix(b, "balance")


# ---------------------------------------------------------------------------
# negotiation
# ---------------------------------------------------------------------------
def test_negotiate_example_basin(basin):
    out = negotiate(basin)
    assert out["rule"] == "talmud"
    # estate = natural flow minus the in-stream flow that must still reach the sea (Delta's 1 500),
    # NOT minus the sum of the three requirements (7 000): upstream in-stream flows are not withdrawals
    assert out["estate"] == pytest.approx(28_000.0 - 1_500.0)
    assert out["estate"] == pytest.approx(bankruptcy_estate(basin)["estate"])
    assert out["natural_flow_mm3"] == pytest.approx(28_000.0)
    assert out["environmental_flow_mm3"] == pytest.approx(7_000.0)  # sum of the requirements, information only
    assert out["terminal_environmental_flow_mm3"] == pytest.approx(1_500.0)
    assert out["environmental_reserve_mm3"] == pytest.approx(1_500.0)
    assert out["claims"] == ALLOC
    assert list(out["proposal"]) == NAMES
    assert sum(out["proposal"].values()) == pytest.approx(min(out["estate"], sum(out["claims"].values())))
    assert out["total_awarded_mm3"] == pytest.approx(sum(out["proposal"].values()))
    assert out["proposal"] == pytest.approx(apply_rule("talmud", out["estate"], ALLOC))
    # Talmud with E = 26 500 > C/2 = 13 750: c/2 + CEL(12 750, c/2) = c/2 + (c/2 - 1000/3)
    assert out["proposal"] == pytest.approx({"Highland": 6_500.0 / 3, "Midland": 26_000.0 / 3, "Delta": 47_000.0 / 3})
    for name in NAMES:
        assert 0.0 <= out["proposal"][name] <= ALLOC[name] + 1e-9
    assert isinstance(out["zopa"], bool)
    assert set(out["acceptable"]) == set(NAMES)
    assert all(isinstance(v, bool) for v in out["acceptable"].values())
    assert out["zopa"] == all(out["acceptable"].values())
    unilateral = route_basin(basin, entitlements={n: None for n in NAMES})
    assert out["batna"] == pytest.approx(unilateral.withdrawals_by_riparian())
    for name in NAMES:
        assert out["acceptable"][name] == (out["proposal"][name] >= out["batna"][name] - 1e-6)
    # at mean flow the upstream riparians now accept; Delta's unilateral 16 300 exceeds its 16 000 claim
    assert out["acceptable"] == {"Highland": True, "Midland": True, "Delta": False} and out["zopa"] is False
    assert out["satisfaction"] == pytest.approx(satisfaction(out["proposal"], ALLOC))
    assert 0.0 <= out["gini"] <= 1.0 and out["gini"] == pytest.approx(gini(out["proposal"]))
    assert 0.0 <= out["gini_satisfaction"] <= 1.0
    routed = route_basin(basin, entitlements=out["proposal"])
    assert out["routed_withdrawals"] == pytest.approx(routed.withdrawals_by_riparian())
    assert out["outflow_to_sea_mm3"] == pytest.approx(routed.outflow_to_sea_mm3)
    for name in NAMES:
        assert out["routed_withdrawals"][name] <= out["proposal"][name] + 1e-9


@pytest.mark.parametrize("rule", list(RULES) + ["adjusted_proportional", "CEA"])
@pytest.mark.parametrize("ff", [0.0, 0.6, 1.0, 1.5])
def test_negotiate_all_rules_sum_to_estate(basin, rule, ff):
    out = negotiate(basin, flow_factor=ff, rule=rule)
    estate = max(28_000.0 * ff - 1_500.0, 0.0)  # natural flow minus the terminal in-stream requirement
    assert out["estate"] == pytest.approx(estate)
    assert out["environmental_reserve_mm3"] == pytest.approx(min(1_500.0, 28_000.0 * ff))
    assert sum(out["proposal"].values()) == pytest.approx(min(estate, sum(ALLOC.values())))
    for name in NAMES:
        assert -1e-9 <= out["proposal"][name] <= ALLOC[name] + 1e-9
    assert isinstance(out["zopa"], bool)


def test_negotiate_zero_flow(basin):
    out = negotiate(basin, flow_factor=0.0)
    assert out["estate"] == 0.0 and out["environmental_reserve_mm3"] == 0.0
    assert all(v == 0.0 for v in out["proposal"].values())
    assert all(v == 1.0 or v == pytest.approx(0.0) for v in out["satisfaction"].values())


def test_negotiate_overrides(basin):
    # claims: demand when there is no treaty entitlement
    b = basin.copy()
    for r in b:
        r.treaty_allocation_mm3 = None
    out = negotiate(b, rule="proportional")
    assert out["claims"] == pytest.approx({r.name: r.demand.total_withdrawal() for r in b})
    # partial claims override keeps the others
    out = negotiate(basin, claims={"Delta": 10_000.0})
    assert out["claims"] == {"Highland": 2_500.0, "Midland": 9_000.0, "Delta": 10_000.0}
    # a generous BATNA override makes the proposal acceptable to everyone
    out = negotiate(basin, batna={n: 0.0 for n in NAMES})
    assert out["zopa"] is True and all(out["acceptable"].values())
    # a BATNA above the claim can never be met
    out = negotiate(basin, batna={"Highland": 1e9})
    assert out["acceptable"]["Highland"] is False and out["zopa"] is False
    # explicit estate: the basin's own reserve is still reported
    out = negotiate(basin, estate=100.0)
    assert out["estate"] == 100.0 and sum(out["proposal"].values()) == pytest.approx(100.0)
    assert out["environmental_reserve_mm3"] == pytest.approx(1_500.0)
    big = negotiate(basin, estate=1e6)
    assert big["proposal"] == pytest.approx(ALLOC)  # every claim honoured
    assert big["acceptable"]["Highland"] is True and big["acceptable"]["Midland"] is True
    # Delta's unilateral withdrawal (16 300) exceeds its 16 000 claim, so no ZOPA
    assert big["acceptable"]["Delta"] is False and big["zopa"] is False
    # tolerance
    tight = negotiate(basin, batna={n: negotiate(basin)["proposal"][n] + 0.5 for n in NAMES}, tol=1.0)
    assert tight["zopa"] is True


def test_negotiate_validation(basin):
    with pytest.raises(ValueError):
        negotiate(basin, rule="robin_hood")
    with pytest.raises(ValueError):
        negotiate(basin, flow_factor=-1.0)
    with pytest.raises(ValueError):
        negotiate(basin, claims={"Nowhere": 1.0})
    with pytest.raises(ValueError):
        negotiate(basin, claims={"Delta": -1.0})
    with pytest.raises(ValueError):
        negotiate(basin, batna={"Nowhere": 1.0})
    with pytest.raises(ValueError):
        negotiate(basin, batna={"Delta": math.nan})
    with pytest.raises(ValueError):
        negotiate(basin, estate=-5.0)
    with pytest.raises(ValueError):
        negotiate(basin, tol=-1.0)
    with pytest.raises(ValueError):
        negotiate(basin, claims=[1, 2, 3])
    with pytest.raises(ValueError):
        negotiate("basin")


def test_negotiate_empty_and_single_basins():
    empty = Basin("empty", [], headwater_inflow_mm3=100.0)
    out = negotiate(empty)
    assert out["proposal"] == {} and out["zopa"] is True and out["gini"] == 0.0
    assert out["estate"] == 100.0 and out["terminal_environmental_flow_mm3"] == 0.0
    single = Basin("one", [make_riparian("A", local=100.0, mun=30.0, env=20.0)])
    out = negotiate(single, rule="cea")
    assert out["estate"] == pytest.approx(100.0 - 20.0)  # local 100 - env 20 (no headwater; the only reach is terminal)
    assert out["proposal"] == {"A": 30.0}
    assert out["batna"]["A"] == pytest.approx(30.0)
    assert out["zopa"] is True


# ---------------------------------------------------------------------------
# bankruptcy estate
# ---------------------------------------------------------------------------
def test_bankruptcy_estate_example_basin_closed_form(basin):
    info = bankruptcy_estate(basin)
    assert info["natural_flow_mm3"] == pytest.approx(28_000.0)
    assert info["environmental_flow_mm3"] == pytest.approx(7_000.0)  # sum of the requirements, information only
    assert info["terminal_environmental_flow_mm3"] == 1_500.0
    assert info["terminal_reach"] == "Delta"
    assert info["estate"] == pytest.approx(26_500.0)  # not 21 000: upstream requirements are not withdrawals
    assert info["environmental_reserve_mm3"] == pytest.approx(1_500.0)
    assert info["headroom_mm3"] == pytest.approx({"Highland": 17_000.0, "Midland": 24_500.0, "Delta": 26_500.0})
    assert info["infeasible_reaches"] == []
    flows = natural_flows(basin)
    for r in basin:
        assert info["headroom_mm3"][r.name] == pytest.approx(flows[r.name] - r.demand.environmental)
    dry = bankruptcy_estate(basin, 0.6)
    assert dry["estate"] == pytest.approx(16_800.0 - 1_500.0)
    assert dry["environmental_reserve_mm3"] == pytest.approx(1_500.0)
    # a drought so deep that Highland cannot meet its own requirement even with zero use:
    # it is reported, and the estate is still the terminal headroom
    deep = bankruptcy_estate(basin, 0.1)
    assert deep["estate"] == pytest.approx(2_800.0 - 1_500.0)
    assert deep["infeasible_reaches"] == ["Highland"]
    assert deep["headroom_mm3"]["Highland"] == pytest.approx(-1_000.0)


def test_bankruptcy_estate_does_not_double_count_in_stream_flows(basin):
    """Water kept in the river at an upstream outlet is still withdrawable below it (review finding)."""
    base = bankruptcy_estate(basin)["estate"]
    # raising satisfiable upstream requirements does not shrink the estate ...
    b = basin.copy()
    b.riparian("Highland").demand.environmental = 12_000.0
    b.riparian("Midland").demand.environmental = 9_000.0
    raised = bankruptcy_estate(b)
    assert raised["estate"] == pytest.approx(base)
    assert raised["environmental_flow_mm3"] == pytest.approx(22_500.0)
    assert raised["headroom_mm3"] == pytest.approx({"Highland": 8_000.0, "Midland": 18_000.0, "Delta": 26_500.0})
    # ... and the physical routing confirms that this much can be consumed while every requirement holds:
    # consume the headroom in Highland, the rest of the cumulative headroom in Midland, the remainder in Delta
    plan = {"Highland": 8_000.0, "Midland": 10_000.0, "Delta": 8_500.0}
    assert sum(plan.values()) == pytest.approx(raised["estate"])
    for name, volume in plan.items():
        b.riparian(name).demand.municipal = volume
        b.riparian(name).demand.industrial = b.riparian(name).demand.agricultural = b.riparian(name).demand.energy = 0.0
        b.riparian(name).demand.consumption_fraction[Sector.MUNICIPAL] = 1.0
        b.riparian(name).groundwater_abstraction_mm3 = 0.0
        b.riparian(name).energy.desalination_capacity_mm3 = 0.0
        b.riparian(name).reservoir_storage_mm3 = b.riparian(name).reservoir_capacity_mm3 = 0.0
    bal = route_basin(b, entitlements={name: None for name in b.names()})  # no treaty caps
    assert bal.total_consumption() == pytest.approx(raised["estate"])
    assert bal.env_flow_met_share() == 1.0 and bal.outflow_to_sea_mm3 == pytest.approx(1_500.0)
    # the old 'sum of requirements' estate would have withheld 22 500 of the 28 000 Mm3
    assert 28_000.0 - 22_500.0 < raised["estate"]
    # whereas the terminal requirement is withheld one for one
    c = basin.copy()
    c.riparian("Delta").demand.environmental = 4_000.0
    assert bankruptcy_estate(c)["estate"] == pytest.approx(base - 2_500.0)
    assert bankruptcy_estate(c)["environmental_reserve_mm3"] == pytest.approx(4_000.0)


def test_bankruptcy_estate_matches_lp_maximum_consumption():
    """Closed form == LP optimum: max total consumption honouring every satisfiable in-stream requirement."""
    from scipy.optimize import linprog

    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(80):
        n = int(rng.integers(1, 6))
        rips = [
            make_riparian(chr(65 + i), local=float(rng.uniform(0, 100)),
                          env=float(rng.uniform(0, 150)) if rng.random() < 0.8 else 0.0)
            for i in range(n)
        ]
        b = Basin("rand", rips, headwater_inflow_mm3=float(rng.uniform(0, 100)))
        info = bankruptcy_estate(b)
        q = np.array(list(natural_flows(b).values()))
        env = np.array([r.demand.environmental for r in rips])
        headroom = q - env
        assert info["headroom_mm3"] == pytest.approx({r.name: h for r, h in zip(rips, headroom)})
        assert info["infeasible_reaches"] == [r.name for r, h in zip(rips, headroom) if h < 0]
        if headroom[-1] < 0:  # terminal requirement unsatisfiable even at zero use: nothing to divide
            assert info["estate"] == 0.0
            continue
        rows, ub, bounds = [], [], []
        for r in range(n):
            row = np.zeros(n)
            row[: r + 1] = 1.0
            if headroom[r] >= 0:
                rows.append(row)
                ub.append(headroom[r])
                bounds.append((0, None))
            else:  # a reach that cannot meet its own requirement consumes nothing (route_basin's cap)
                bounds.append((0, 0))
        res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(ub), bounds=bounds, method="highs")
        assert res.status == 0
        assert info["estate"] == pytest.approx(-res.fun, abs=1e-6)
        checked += 1
    assert checked > 20


def test_bankruptcy_estate_properties_and_edge_cases(basin):
    estates = [bankruptcy_estate(basin, ff)["estate"] for ff in np.linspace(0.0, 2.0, 21)]
    assert all(e >= 0.0 for e in estates) and estates == sorted(estates)  # monotone in the flow
    for ff in (0.0, 0.3, 1.0, 1.7):
        info = bankruptcy_estate(basin, ff)
        assert info["estate"] <= info["natural_flow_mm3"] + 1e-9
        assert info["estate"] + info["environmental_reserve_mm3"] == pytest.approx(info["natural_flow_mm3"])
        assert info["estate"] == pytest.approx(max(info["natural_flow_mm3"] - 1_500.0, 0.0))
        assert info["environmental_flow_mm3"] == pytest.approx(7_000.0)  # requirements do not scale with the flow
    zero = bankruptcy_estate(basin, 0.0)
    assert zero["estate"] == 0.0 and zero["environmental_reserve_mm3"] == 0.0 and zero["natural_flow_mm3"] == 0.0
    assert zero["infeasible_reaches"] == NAMES
    # terminal requirement larger than the whole flow: nothing to divide, everything is reserved
    big = Basin("big", [make_riparian("A", local=100.0, env=0.0), make_riparian("B", local=0.0, env=1_000.0)])
    info = bankruptcy_estate(big)
    assert info["estate"] == 0.0 and info["infeasible_reaches"] == ["B"]
    assert info["environmental_reserve_mm3"] == pytest.approx(100.0)  # the whole natural flow (no headwater)
    # no requirement anywhere: the whole natural flow is divisible
    assert bankruptcy_estate(chain(3))["estate"] == pytest.approx(500.0 + 3 * 1_000.0)
    # a single riparian: its own requirement is the terminal one
    single = Basin("one", [make_riparian("A", local=100.0, env=20.0)])
    assert bankruptcy_estate(single)["estate"] == pytest.approx(80.0)
    # no riparians: the headwater flow with nothing withheld
    empty = bankruptcy_estate(Basin("empty", [], headwater_inflow_mm3=100.0))
    assert empty["estate"] == 100.0 and empty["terminal_reach"] is None and empty["headroom_mm3"] == {}
    assert empty["terminal_environmental_flow_mm3"] == 0.0 and empty["infeasible_reaches"] == []
    # validation
    for bad in (-1.0, math.inf, math.nan, "1", None):
        with pytest.raises(ValueError):
            bankruptcy_estate(basin, bad)
    with pytest.raises(ValueError):
        bankruptcy_estate("basin")
    neg = basin.copy()
    neg.riparian("Delta").demand.environmental = -1.0
    with pytest.raises(ValueError):
        bankruptcy_estate(neg)
    neg_flow = basin.copy()
    neg_flow.riparian("Midland").local_inflow_mm3 = -5.0
    with pytest.raises(ValueError):
        bankruptcy_estate(neg_flow)
    # no mutation and an independent result
    snapshot = copy.deepcopy(basin)
    info = bankruptcy_estate(basin)
    info["headroom_mm3"]["Delta"] = 0.0
    info["infeasible_reaches"].append("x")
    assert basin == snapshot


def test_cooperative_rules_no_longer_manufacture_scarcity(basin):
    """Review finding: at mean flow the cooperative awards must not cap riparians far below their
    demand while far more than the terminal requirement flows to the sea."""
    out = negotiate(basin)  # talmud, mean flow
    routed = route_basin(basin, entitlements=out["proposal"])
    river_demand = {r.name: r.total_demand() - r.non_river_supply_mm3 for r in routed.reaches}
    # upstream riparians receive awards above their river demand: uncapped, supply ratio 1
    assert out["proposal"]["Highland"] > river_demand["Highland"]
    assert out["proposal"]["Midland"] > river_demand["Midland"]
    assert routed.reach("Highland").supply_ratio() == pytest.approx(1.0)
    assert routed.reach("Midland").supply_ratio() == pytest.approx(1.0)
    assert routed.reach("Delta").supply_ratio() > 0.95
    # consumption of the routed awards never exceeds the (consumptive) estate, every requirement is met
    assert routed.total_consumption() <= out["estate"] + 1e-6
    assert routed.env_flow_met_share() == 1.0
    assert routed.outflow_to_sea_mm3 >= basin.riparian("Delta").demand.environmental
    # the old estate (natural flow - sum of requirements = 21 000) capped everybody below 90 % of demand
    old = route_basin(basin, entitlements=apply_rule("talmud", 28_000.0 - 7_000.0, ALLOC))
    assert all(r.supply_ratio() < 0.9 for r in old.reaches)
    assert old.outflow_to_sea_mm3 > routed.outflow_to_sea_mm3
    for rule, row in compare_allocation_rules(basin).items():
        assert row["supply_ratio"] > 0.95, rule
        assert row["env_flow_met_share"] == 1.0, rule
    # in a 40 % drought Delta's Talmud award more than doubles against the old estate
    dry = negotiate(basin, 0.6)
    assert dry["estate"] == pytest.approx(15_300.0)
    assert dry["proposal"] == pytest.approx({"Highland": 1_250.0, "Midland": 4_500.0, "Delta": 9_550.0})
    assert apply_rule("talmud", 28_000.0 * 0.6 - 7_000.0, ALLOC)["Delta"] == pytest.approx(4_275.0)


# ---------------------------------------------------------------------------
# rule comparison
# ---------------------------------------------------------------------------
def test_compare_allocation_rules_covers_all_rules(basin):
    out = compare_allocation_rules(basin)
    assert list(out) == list(RULES)
    estate = 26_500.0  # 28 000 natural flow - 1 500 terminal in-stream requirement
    total_claims = sum(ALLOC.values())
    for rule, row in out.items():
        assert row["estate"] == pytest.approx(estate) and row["claims"] == ALLOC
        assert list(row["awards"]) == NAMES
        assert sum(row["awards"].values()) == pytest.approx(min(estate, total_claims))
        assert row["total_awarded_mm3"] == pytest.approx(sum(row["awards"].values()))
        assert row["awards"] == pytest.approx(apply_rule(rule, estate, ALLOC))
        for name in NAMES:
            assert -1e-9 <= row["awards"][name] <= ALLOC[name] + 1e-9
        assert 0.0 <= row["gini"] <= 1.0 and 0.0 <= row["gini_satisfaction"] <= 1.0
        assert row["min_satisfaction"] == pytest.approx(min(row["satisfaction"].values()))
        assert 0.0 <= row["min_satisfaction"] <= 1.0
        bal = route_basin(basin, entitlements=row["awards"])
        assert row["outflow_to_sea_mm3"] == pytest.approx(bal.outflow_to_sea_mm3)
        assert row["outflow_to_sea_mm3"] >= 0.0
        assert row["withdrawals"] == pytest.approx(bal.withdrawals_by_riparian())
        assert row["total_withdrawal_mm3"] == pytest.approx(bal.total_withdrawal())
        assert 0.0 <= row["supply_ratio"] <= 1.0 and 0.0 <= row["env_flow_met_share"] <= 1.0
        for name in NAMES:
            assert row["withdrawals"][name] <= row["awards"][name] + 1e-9
    assert out["equal"]["awards"] == pytest.approx(out["cea"]["awards"])
    # proportional is the only rule with equal satisfaction (gini of satisfaction 0)
    assert out["proportional"]["gini_satisfaction"] == pytest.approx(0.0)
    assert out["cel"]["min_satisfaction"] < out["proportional"]["min_satisfaction"]


def test_compare_allocation_rules_subset_and_options(basin):
    out = compare_allocation_rules(basin, flow_factor=0.6, rules=["talmud", "proportional"], claims={"Delta": 12_000.0})
    assert list(out) == ["talmud", "proportional"]
    assert out["talmud"]["estate"] == pytest.approx(28_000.0 * 0.6 - 1_500.0)
    assert out["talmud"]["claims"]["Delta"] == 12_000.0
    assert out["talmud"]["awards"] == pytest.approx(negotiate(basin, 0.6, "talmud", claims={"Delta": 12_000.0})["proposal"])
    fixed = compare_allocation_rules(basin, rules=["cea"], estate=1_000.0)
    assert sum(fixed["cea"]["awards"].values()) == pytest.approx(1_000.0)
    with pytest.raises(ValueError):
        compare_allocation_rules(basin, rules=["nope"])
    with pytest.raises(ValueError):
        compare_allocation_rules(basin, rules=["cea", "cea"])
    with pytest.raises(ValueError):
        compare_allocation_rules(basin, flow_factor=math.inf)
    empty = compare_allocation_rules(Basin("empty", []))
    assert all(row["awards"] == {} and row["min_satisfaction"] == 1.0 for row in empty.values())


# ---------------------------------------------------------------------------
# no mutation
# ---------------------------------------------------------------------------
def test_diplomacy_functions_do_not_mutate_inputs(basin):
    snapshot = copy.deepcopy(basin)
    bal = route_basin(basin)
    bal_snapshot = copy.deepcopy(bal)
    treaty = full_treaty()
    treaty_snapshot = copy.deepcopy(treaty)
    events = copy.deepcopy(EVENTS)
    for r in basin:
        hydro_hegemony(r, basin, weights={"material": 2.0})
    power_asymmetry(basin)
    cooperation_index(events, window_years=5)
    cooperation_intensity(events, window_years=5)
    conflict_intensity(events)
    bar_event_summary(events)
    treaty_resilience(treaty)
    treaty_compliance(treaty, bal)
    water_dependency(basin)
    water_dependency(basin, bal)
    conflict_risk_index(basin, bal, treaty, events, climate_cv=0.3)
    benefit_sharing_matrix(basin, bal, route_basin(basin, 0.8), treaty=treaty, events=events)
    negotiate(basin, 0.7, "cel", claims={"Delta": 1.0}, batna={"Delta": 0.0})
    compare_allocation_rules(basin, 0.9)
    bankruptcy_estate(basin, 0.4)
    treaty_from_basin(basin)
    assert_unchanged(basin, snapshot)
    assert bal == bal_snapshot
    assert treaty == treaty_snapshot
    assert events == EVENTS


def test_results_are_independent_copies(basin):
    out = negotiate(basin)
    out["claims"]["Delta"] = 0.0
    assert basin.riparian("Delta").treaty_allocation_mm3 == 16_000.0
    t = bare_treaty()
    ent = t.entitlements()
    ent["Delta"] = 0.0
    assert t.allocations_mm3["Delta"] == 16_000.0
