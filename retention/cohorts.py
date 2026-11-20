"""Weekly cohort retention and its decomposition.

The business question is why week-to-week retention moved. A raw week-over-week
comparison of repurchase rates can move for three different reasons:

1. **Mix**: the week's first-purchase cohort has a different channel or
   first-purchase scenario mix.
2. **Within-stratum movement**: a channel or scenario genuinely retains better
   or worse than it used to.
3. **Observation status**: a recent cohort has not yet had a full window, or a
   purchase inside the window is still waiting on after-sales maturity.

The decomposition below is a standard rate-difference decomposition:

    rate_t - rate_base = sum_s (w_ts - w_bs) * p_bs   [mix]
                       + sum_s  w_ts       * (p_ts - p_bs)  [within]
                       + interaction

with an explicit interaction term, so the three pieces add back to the observed
difference. Weeks that are not mature are reported separately and excluded from
the trend test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .orders import CohortMember
from .stats import heterogeneity_test, proportion, trend_test, wilson_interval


@dataclass(frozen=True, slots=True)
class Stratum:
    channel: str
    context: str

    def key(self, dimension: str) -> tuple[str, ...]:
        if dimension == "channel":
            return (self.channel,)
        if dimension == "context":
            return (self.context,)
        return (self.channel, self.context)


@dataclass(frozen=True, slots=True)
class WeekCohort:
    week: str
    members: tuple[CohortMember, ...]
    mature: tuple[CohortMember, ...]
    pending: tuple[CohortMember, ...]
    repurchases: int

    @property
    def size(self) -> int:
        return len(self.mature)

    @property
    def rate(self) -> float:
        return proportion(self.repurchases, self.size)

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.repurchases, self.size)

    def strata(self, dimension: str = "both") -> dict[Stratum, list[CohortMember]]:
        out: dict[Stratum, list[CohortMember]] = defaultdict(list)
        for member in self.mature:
            out[Stratum(member.first.channel, member.first.context)].append(member)
        if dimension == "both":
            return out
        merged: dict[tuple[str, ...], list[CohortMember]] = defaultdict(list)
        for stratum, rows in out.items():
            merged[stratum.key(dimension)].extend(rows)
        return {
            Stratum(*key, *([""] * (2 - len(key)))): rows
            for key, rows in merged.items()
        }


@dataclass(frozen=True, slots=True)
class MixEffect:
    stratum: Stratum
    base_weight: float
    week_weight: float
    base_rate: float
    week_rate: float
    mix_contribution: float
    within_contribution: float
    interaction_contribution: float
    week_successes: int
    week_size: int


@dataclass(frozen=True, slots=True)
class WeekComparison:
    week: str
    baseline: str
    rate: float
    baseline_rate: float
    difference: float
    mix: float
    within: float
    interaction: float
    detail: tuple[MixEffect, ...]
    common_support_share: float
    excluded_weight: float
    outside_support_difference: float
    within_support_difference: float


def weekly_cohorts(members: Sequence[CohortMember]) -> list[WeekCohort]:
    grouped: dict[str, list[CohortMember]] = defaultdict(list)
    for member in members:
        grouped[member.week].append(member)
    weeks: list[WeekCohort] = []
    for week in sorted(grouped):
        all_members = grouped[week]
        mature = tuple(m for m in all_members if m.status == "mature")
        pending = tuple(m for m in all_members if m.status != "mature")
        weeks.append(
            WeekCohort(
                week=week,
                members=tuple(all_members),
                mature=mature,
                pending=pending,
                repurchases=sum(1 for m in mature if m.repurchased),
            )
        )
    return weeks


def trend(weeks: Sequence[WeekCohort]) -> dict[str, object]:
    """Heterogeneity and trend tests across fully mature weeks."""
    mature = [w for w in weeks if w.size >= 1]
    pairs = [(w.repurchases, w.size) for w in mature]
    chi2, p_value = heterogeneity_test(pairs)
    indexed = [(index, w.repurchases, w.size) for index, w in enumerate(mature)]
    z, trend_p = trend_test(indexed)
    return {
        "weeks": len(mature),
        "heterogeneity_chi2": chi2,
        "heterogeneity_p": p_value,
        "trend_z": z,
        "trend_p": trend_p,
        "min_rate": min((w.rate for w in mature), default=float("nan")),
        "max_rate": max((w.rate for w in mature), default=float("nan")),
    }


def decompose(week: WeekCohort, baseline: WeekCohort, *, dimension: str = "both") -> WeekComparison:
    """Decompose ``week.rate - baseline.rate`` on the strata both weeks share.

    ``dimension`` selects the composition control: ``channel``, ``context`` or
    both. A stratum that exists in only one of the two weeks has no
    counterfactual rate, so it cannot enter a mix/within split. Those units are
    set aside and their total contribution is reported as
    ``outside_support_difference``; the three terms on common support plus that
    residual add back exactly to the observed difference.
    """
    week_strata = week.strata(dimension)
    base_strata = baseline.strata(dimension)
    all_keys = set(week_strata) | set(base_strata)
    common = sorted(
        (key for key in all_keys if week_strata.get(key) and base_strata.get(key)),
        key=lambda s: s.key(dimension),
    )
    week_n, base_n = week.size, baseline.size
    week_common = sum(len(week_strata[k]) for k in common)
    base_common = sum(len(base_strata[k]) for k in common)
    week_common_success = sum(
        1 for k in common for m in week_strata[k] if m.repurchased
    )
    base_common_success = sum(
        1 for k in common for m in base_strata[k] if m.repurchased
    )
    week_common_rate = week_common_success / week_common if week_common else float("nan")
    base_common_rate = base_common_success / base_common if base_common else float("nan")
    week_rate = week.repurchases / week_n if week_n else float("nan")
    base_rate = baseline.repurchases / base_n if base_n else float("nan")

    details: list[MixEffect] = []
    mix_total = within_total = interaction_total = 0.0
    for key in common:
        w_rows = week_strata[key]
        b_rows = base_strata[key]
        w_weight = len(w_rows) / week_common if week_common else 0.0
        b_weight = len(b_rows) / base_common if base_common else 0.0
        b_stratum_rate = sum(1 for m in b_rows if m.repurchased) / len(b_rows)
        w_stratum_rate = sum(1 for m in w_rows if m.repurchased) / len(w_rows)
        mix = (w_weight - b_weight) * b_stratum_rate
        within = b_weight * (w_stratum_rate - b_stratum_rate)
        interaction = (w_weight - b_weight) * (w_stratum_rate - b_stratum_rate)
        mix_total += mix
        within_total += within
        interaction_total += interaction
        details.append(
            MixEffect(
                stratum=key,
                base_weight=b_weight,
                week_weight=w_weight,
                base_rate=b_stratum_rate,
                week_rate=w_stratum_rate,
                mix_contribution=mix,
                within_contribution=within,
                interaction_contribution=interaction,
                week_successes=sum(1 for m in w_rows if m.repurchased),
                week_size=len(w_rows),
            )
        )

    # Exact remainder outside common support: the week-only units at the week
    # rate, plus the missing baseline units evaluated at the baseline rate.
    within_support_difference = week_common_rate - base_common_rate
    outside = float("nan")
    if week_n and base_n:
        outside = (week_rate - week_common_rate) + (base_common_rate - base_rate)
    return WeekComparison(
        week=week.week,
        baseline=baseline.week,
        rate=week_rate,
        baseline_rate=base_rate,
        difference=week_rate - base_rate,
        mix=mix_total,
        within=within_total,
        interaction=interaction_total,
        detail=tuple(details),
        common_support_share=(week_common / week_n) if week_n else float("nan"),
        excluded_weight=((week_n - week_common) / week_n) if week_n else float("nan"),
        outside_support_difference=outside,
        within_support_difference=within_support_difference,
    )


def weekly_decomposition(
    members: Sequence[CohortMember],
    *,
    baseline: str | None = None,
    alternatives: Sequence[str] = ("channel", "context"),
) -> dict[str, object]:
    """Full weekly picture with maturity status and mix/within decomposition."""
    weeks = weekly_cohorts(members)
    mature_weeks = [w for w in weeks if w.size > 0]
    if baseline is None:
        baseline_week = mature_weeks[0].week if mature_weeks else None
    else:
        baseline_week = baseline
    baseline_obj = next((w for w in weeks if w.week == baseline_week), None)
    comparisons = []
    if baseline_obj is not None:
        for week in weeks:
            if week.week == baseline_obj.week or week.size == 0:
                continue
            comparisons.append(decompose(week, baseline_obj))
    alternative_comparisons: dict[str, list[WeekComparison]] = {}
    if baseline_obj is not None:
        for dimension in alternatives:
            alternative_comparisons[dimension] = [
                decompose(week, baseline_obj, dimension=dimension)
                for week in weeks
                if week.week != baseline_obj.week and week.size > 0
            ]
    return {
        "weeks": weeks,
        "baseline": baseline_week,
        "comparisons": comparisons,
        "alternatives": alternative_comparisons,
        "trend": trend(mature_weeks),
        "pending_units": sum(len(w.pending) for w in weeks),
    }


def format_activity_table(activity: dict[str, dict[str, int]]) -> str:
    """Weekly order table separating new payer units from repeat orders."""
    lines = [
        "| 周 | 主动订单 | 其中新付款主体 | 其中回购买单 | 礼赠场景订单 | 未确认身份订单 | 毛利 |",
        "|---|---|---|---|---|---|---|",
    ]
    for week, counts in activity.items():
        lines.append(
            f"| {week} | {counts.get('orders', 0)} | {counts.get('new_units', 0)} | "
            f"{counts.get('repeat_orders', 0)} | {counts.get('gift_orders', 0)} | "
            f"{counts.get('unconfirmed_orders', 0)} | {counts.get('margin', 0) / 100:,.2f} 元 |"
        )
    return "\n".join(lines)


def format_week_table(weeks: Sequence[WeekCohort]) -> str:
    lines = ["| 首购周 | 成熟 n | 复购 n | 复购率 | 95% 区间 | 未成熟 n |", "|---|---|---|---|---|---|"]
    for week in weeks:
        if week.size == 0:
            lines.append(f"| {week.week} | 0 | — | 待定 | — | {len(week.pending)} |")
            continue
        low, high = week.interval
        lines.append(
            f"| {week.week} | {week.size} | {week.repurchases} | "
            f"{week.rate * 100:.1f}% | {low * 100:.1f}%–{high * 100:.1f}% | {len(week.pending)} |"
        )
    return "\n".join(lines)
