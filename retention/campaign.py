"""The next-week win-back plan, its exclusions, and late-binding reconciliation.

Operating rules encoded here:

* Only payer units with a confirmed identity can be contacted. Anonymous
  visitors can be *estimated* as a group but never listed as a contact.
* One unit is one contact even when it holds an App account and a storefront
  account. Sending one coupon per account is a duplicate send.
* A unit that bought inside the repurchase window has already come back by
  itself and is not "recoverable this week".
* An issued, unredeemed, unexpired coupon still reserves budget and may occupy
  support time. The unit holding it is not contacted again.
* The policy's holdout share is a commitment about *not sending*, so the plan
  is stratified and a comparable held-out group is kept inside each stratum.
* Because no positive incremental effect was demonstrated historically, the
  plan reports expected value honestly instead of multiplying a null lift by
  the segment size.
* A late binding never revokes a redeemed benefit. It does collapse the
  household's unsent contacts to one, remove that household from the
  coupon/no-coupon contrast, and report the budget and comparison scope that
  the duplicate consumed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import Dataset
from .experiments import Estimate, OfferStudy
from .identity import IdentityMap
from .orders import OrderBook
from .stats import wilson_interval


@dataclass(frozen=True, slots=True)
class CampaignRequest:
    """What operations actually submits."""

    start: datetime
    days: int
    budget_cents: int
    coupon_cents: int
    valid_until: datetime
    max_contacts_per_day: int
    preserve_holdout_share: float
    segments: tuple[str, ...] = ("自用", "礼赠", "团购")
    min_days_since_last: int = 31
    exclude_never_couponed_only: bool = False


@dataclass(frozen=True, slots=True)
class Candidate:
    unit: str
    handles: tuple[str, ...]
    household: str
    channel: str
    first_context: str
    last_context: str
    days_since_last: int
    purchases: int
    mean_margin: float
    never_couponed: bool
    stratum: str
    comparison_lost: bool = False


@dataclass(frozen=True, slots=True)
class Exclusion:
    unit: str
    reason: str


@dataclass(frozen=True, slots=True)
class Allocation:
    unit: str
    handles: tuple[str, ...]
    stratum: str
    role: str  # "coupon" | "holdout"
    never_couponed: bool
    comparison_eligible: bool = True


@dataclass
class CampaignPlan:
    request: CampaignRequest
    candidates: tuple[Candidate, ...]
    exclusions: tuple[Exclusion, ...]
    allocations: tuple[Allocation, ...]
    outstanding_coupons: tuple[dict, ...]
    daily_slots: tuple[tuple[str, int], ...]
    expected: dict[str, object]
    anonymous_estimate: dict[str, object]
    identity_coverage: dict[str, object]
    household_dedup_units: tuple[str, ...]
    notes: tuple[str, ...]

    # -- derived ---------------------------------------------------------
    @property
    def coupon_units(self) -> tuple[Allocation, ...]:
        return tuple(a for a in self.allocations if a.role == "coupon")

    @property
    def holdout_units(self) -> tuple[Allocation, ...]:
        return tuple(a for a in self.allocations if a.role == "holdout")

    @property
    def comparison_allocations(self) -> tuple[Allocation, ...]:
        """Allocations that may still serve as a coupon / no-coupon comparison."""
        return tuple(a for a in self.allocations if a.comparison_eligible)

    @property
    def reserved_budget(self) -> int:
        return len(self.coupon_units) * self.request.coupon_cents

    @property
    def carry_in_budget(self) -> int:
        return sum(int(c["cents"]) for c in self.outstanding_coupons)

    @property
    def contact_slots_used(self) -> int:
        """Outbound contact slots. A held-out unit is not contacted."""
        return len(self.coupon_units)

    def to_json(self) -> str:
        payload = {
            "request": {
                **asdict(self.request),
                "start": self.request.start.isoformat(),
                "valid_until": self.request.valid_until.isoformat(),
            },
            "allocations": [asdict(a) for a in self.allocations],
            "exclusions": [asdict(e) for e in self.exclusions],
            "outstanding_coupons": list(self.outstanding_coupons),
            "daily_slots": list(self.daily_slots),
            "expected": _jsonable(self.expected),
            "anonymous_estimate": _jsonable(self.anonymous_estimate),
            "identity_coverage": _jsonable(self.identity_coverage),
            "household_dedup_units": list(self.household_dedup_units),
            "notes": list(self.notes),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _stratum(channel: str, context: str, days: int) -> str:
    band = "31-60" if days <= 60 else "61-90" if days <= 90 else "91-120" if days <= 120 else "120+"
    return f"{channel}|{context}|{band}"


def plan_campaign(
    dataset: Dataset,
    request: CampaignRequest,
    *,
    study: OfferStudy | None = None,
) -> CampaignPlan:
    identity = IdentityMap(dataset, as_of=request.start)
    book = OrderBook(dataset, identity, dataset.policy.repurchase_window_days)
    study = study or OfferStudy(dataset, as_of=dataset.latest_checkpoint)

    outstanding = [
        c
        for c in dataset.coupons
        if c.state == "已发" and c.expires_at > request.start
    ]
    outstanding_by_unit = {identity.unit_key(c.sent_to): c for c in outstanding}
    outstanding_households = {identity.household_key(c.sent_to) for c in outstanding}
    # Households where a late binding destroyed the *historical* comparison.
    # They are reported and contacted as one household going forward; the
    # historical damage is not a reason to refuse service in a new week.
    comparison_lost_households = _late_binding_households(dataset)

    exclusions: list[Exclusion] = []
    candidates: list[Candidate] = []
    units_with_purchases = [
        unit for unit in book.by_unit if book.active_purchases(unit, as_of=request.start)
    ]
    for unit in sorted(units_with_purchases):
        handles = identity.handles_of_unit(unit)
        if not identity.unit_is_confirmed(unit):
            exclusions.append(Exclusion(unit, "身份未确认，仅可作范围估计，不可联系"))
            continue
        if not all(
            identity.is_visible(h)
            and dataset.accounts_by_handle[h].contact_permission == "允许联系"
            for h in handles
        ):
            exclusions.append(Exclusion(unit, "无联系权限或记录在发券时点尚不可见"))
            continue
        household = identity.household_of_unit(unit)
        if unit in outstanding_by_unit:
            exclusions.append(Exclusion(unit, "已持未兑券，仍占预算与客服时段"))
            continue
        if household in outstanding_households:
            exclusions.append(Exclusion(unit, "同户已持未兑券"))
            continue
        last = book.last_purchase(unit, as_of=request.start)
        if last is None:  # pragma: no cover - guarded above
            continue
        days = (request.start - last.paid_at).days
        prior_active = book.active_purchases(unit, as_of=request.start)
        if days < request.min_days_since_last:
            if len(prior_active) >= 2:
                exclusions.append(
                    Exclusion(unit, f"最近 {days} 天内已自行回购，不算本周可挽回")
                )
            else:
                exclusions.append(
                    Exclusion(unit, f"首购仅 {days} 天，尚未渡过观察期，不按可挽回计")
                )
            continue
        first = book.first_purchase(unit, as_of=request.start)[0]
        if first is None:  # pragma: no cover
            continue
        if first.context not in request.segments:
            exclusions.append(Exclusion(unit, f"首购场景 {first.context} 不在本周客群"))
            continue
        never_couponed = not any(
            identity.unit_key(c.sent_to) == unit for c in dataset.coupons
        )
        if request.exclude_never_couponed_only and not never_couponed:
            exclusions.append(Exclusion(unit, "本周只保留从未接触过的对象"))
            continue
        margins = [e.margin for e in book.active_purchases(unit, as_of=request.start)]
        channel = sorted(
            {dataset.accounts_by_handle[h].channel for h in handles}
        )[0]
        candidates.append(
            Candidate(
                unit=unit,
                handles=handles,
                household=household,
                channel=channel,
                first_context=first.context,
                last_context=last.context,
                days_since_last=days,
                purchases=len(margins),
                mean_margin=float(np.mean(margins)),
                never_couponed=never_couponed,
                stratum=_stratum(channel, first.context, days),
                comparison_lost=household in comparison_lost_households,
            )
        )

    # Capacity and budget. Only an outbound coupon consumes a contact slot; a
    # held-out unit is not contacted and consumes neither capacity nor budget.
    # An unexpired coupon issued earlier still reserves face value inside the
    # campaign window, so it is deducted before new coupons are planned.
    carry_in_cents = sum(c.cents for c in outstanding)
    available_budget = max(0, request.budget_cents - carry_in_cents)
    total_slots = request.max_contacts_per_day * request.days
    budget_slots = available_budget // request.coupon_cents
    coupon_cap = max(0, min(total_slots, budget_slots))
    # Holdout share is the committed fraction of the contacted-or-held-out
    # pool that receives nothing: holdouts = coupons * s / (1 - s).
    share = min(max(request.preserve_holdout_share, 0.0), 0.95)
    holdout_target = int(np.ceil(coupon_cap * share / (1 - share))) if share > 0 else 0

    allocations, household_duplicates = _allocate(candidates, coupon_cap, holdout_target)
    for household, units in sorted(household_duplicates.items()):
        for unit in units:
            exclusions.append(Exclusion(unit, "同户多个付款主体，本周只保留一个联系对象"))
    touched_households = {
        candidate.household
        for candidate in candidates
        if candidate.household in household_duplicates
    }

    # Daily schedule: coupon sends spread across the campaign days.
    daily: list[tuple[str, int]] = []
    coupons = [a for a in allocations if a.role == "coupon"]
    remaining = list(coupons)
    per_day = request.max_contacts_per_day
    current = request.start
    while remaining:
        chunk, remaining = remaining[:per_day], remaining[per_day:]
        daily.append((current.date().isoformat(), len(chunk)))
        current += timedelta(days=1)

    anonymous = _anonymous_estimate(book, identity, request.start)
    coverage = _identity_coverage(dataset, book, identity, request.start)

    estimate = study.primary()
    expected = _expected_value(estimate, allocations, request, study=study)

    notes = [
        "主要历史效应为负且区间跨零，不能用它支持“发券能提升复购”的结论。",
        "计划保留同层可比的不发券对象；券与留量按渠道×首购场景×最近购买间隔分层。",
        "未登录访客只给整体留存范围，不进入任何联系名单。",
        "多平台账户按已确认付款人合并为一个联系对象。",
    ]
    return CampaignPlan(
        request=request,
        candidates=tuple(candidates),
        exclusions=tuple(exclusions),
        allocations=tuple(allocations),
        outstanding_coupons=tuple(
            {
                "coupon": c.coupon,
                "sent_to": c.sent_to,
                "unit": identity.unit_key(c.sent_to),
                "expires_at": c.expires_at.isoformat(),
                "cents": c.cents,
            }
            for c in outstanding
        ),
        daily_slots=tuple(daily),
        expected=expected,
        anonymous_estimate=anonymous,
        identity_coverage=coverage,
        household_dedup_units=tuple(
            unit for units in household_duplicates.values() for unit in units
        ),
        notes=tuple(notes),
    )


def _late_binding_households(dataset: Dataset) -> set[str]:
    """Households where a binding was known only after a coupon was issued."""
    identity = IdentityMap(dataset)
    grouped: dict[str, list] = defaultdict(list)
    for coupon in dataset.coupons:
        grouped[identity.household_key(coupon.sent_to)].append(coupon)
    lost: set[str] = set()
    for household, coupons in grouped.items():
        involved = {c.sent_to for c in coupons}
        bindings = [
            b
            for b in dataset.bindings
            if b.verified and (b.left_handle in involved or b.right_handle in involved)
        ]
        if bindings and any(c.issued_at < min(b.known_at for b in bindings) for c in coupons):
            lost.add(household)
    return lost


def _allocate(
    candidates: Sequence[Candidate], coupon_cap: int, holdout_target: int
) -> tuple[list[Allocation], dict[str, list[str]]]:
    """Stratified allocation, never-contacted and never-couponed first.

    Within each stratum, candidates are ordered by recency band (already in the
    key) and then deterministically by unit, so the split is reproducible and
    not driven by a model whose validity we could not establish.

    A household contributes at most one contact. Two different confirmed payers
    can share a verified household; contacting both would duplicate the coupon
    and break independence, so only the first payer unit is kept and the other
    is returned as a duplicate for reporting.
    """
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.stratum].append(candidate)
    for key in by_stratum:
        by_stratum[key].sort(
            key=lambda c: (c.comparison_lost, not c.never_couponed, c.days_since_last, c.unit)
        )

    allocations: list[Allocation] = []
    households_used: dict[str, str] = {}
    duplicate_units: dict[str, list[str]] = defaultdict(list)
    coupons_left = coupon_cap
    holdouts_left = holdout_target
    # Round-robin over strata so no single stratum absorbs the whole budget.
    order = sorted(by_stratum)
    while coupons_left > 0 or holdouts_left > 0:
        progressed = False
        for key in order:
            pool = by_stratum[key]
            while pool:
                candidate = pool.pop(0)
                household = candidate.household
                if household in households_used:
                    duplicate_units[household].append(candidate.unit)
                    continue
                if coupons_left == 0 and holdouts_left == 0:
                    pool.insert(0, candidate)
                    break
                role = "coupon" if coupons_left > 0 else "holdout"
                allocations.append(
                    Allocation(
                        unit=candidate.unit,
                        handles=candidate.handles,
                        stratum=key,
                        role=role,
                        never_couponed=candidate.never_couponed,
                        comparison_eligible=not candidate.comparison_lost,
                    )
                )
                households_used[household] = candidate.unit
                if role == "coupon":
                    coupons_left -= 1
                else:
                    holdouts_left -= 1
                progressed = True
                break
            if coupons_left == 0 and holdouts_left == 0:
                break
        if not progressed:
            break
    return allocations, dict(duplicate_units)


def _identity_coverage(
    dataset: Dataset, book: OrderBook, identity: IdentityMap, as_of: datetime
) -> dict[str, object]:
    """How much of the business a confirmed-payer view can actually see.

    Unconfirmed accounts cannot be merged into payers, so treating each as a
    payer inflates the denominator. Both counts are reported; only the confirmed
    view drives the contact list.
    """
    active = [e for e in book.events if book.is_known_purchase(e, as_of)]
    confirmed_orders = sum(1 for e in active if e.verified_identity)
    total_margin = sum(e.margin for e in active)
    confirmed_margin = sum(e.margin for e in active if e.verified_identity)
    units = {e.unit for e in active}
    confirmed_units = {u for u in units if identity.unit_is_confirmed(u)}

    members = book.build_cohorts(as_of)
    mature = [m for m in members if m.status == "mature"]
    confirmed_mature = [m for m in mature if m.first.verified_identity]
    return {
        "active_orders": len(active),
        "active_orders_confirmed": confirmed_orders,
        "order_share_confirmed": confirmed_orders / len(active) if active else float("nan"),
        "margin_share_confirmed": confirmed_margin / total_margin if total_margin else float("nan"),
        "units": len(units),
        "confirmed_units": len(confirmed_units),
        "unconfirmed_units": len(units) - len(confirmed_units),
        "cohort_rate_confirmed": (
            sum(1 for m in confirmed_mature if m.repurchased) / len(confirmed_mature)
            if confirmed_mature
            else float("nan")
        ),
        "cohort_rate_confirmed_n": len(confirmed_mature),
        "cohort_rate_all": (
            sum(1 for m in mature if m.repurchased) / len(mature) if mature else float("nan")
        ),
        "cohort_rate_all_n": len(mature),
    }


def _anonymous_estimate(book: OrderBook, identity: IdentityMap, as_of: datetime) -> dict[str, object]:
    units = [
        unit
        for unit in book.by_unit
        if not identity.unit_is_confirmed(unit)
        and book.first_purchase(unit, as_of)[0] is not None
        and book.first_purchase(unit, as_of)[0].channel == "匿名来客"
    ]
    matured = []
    for unit in units:
        first = book.first_purchase(unit, as_of)[0]
        if first.paid_at + timedelta(days=book.window_days) <= as_of:
            matured.append(unit)
    successes = 0
    for unit in matured:
        first = book.first_purchase(unit, as_of)[0]
        later = book.purchases_between(
            unit, first.paid_at, first.paid_at + timedelta(days=book.window_days), as_of=as_of
        )
        successes += bool(later)
    low, high = wilson_interval(successes, len(matured))
    return {
        "units": len(matured),
        "repurchases": successes,
        "point": successes / len(matured) if matured else float("nan"),
        "low": low,
        "high": high,
        "contactable": False,
        "note": "未登录访客只能给出整体留存范围，不能列为可联系对象。",
    }


def _expected_value(
    estimate: Estimate,
    allocations: Sequence[Allocation],
    request: CampaignRequest,
    *,
    study: OfferStudy | None = None,
) -> dict[str, object]:
    coupons = [a for a in allocations if a.role == "coupon"]
    n = len(coupons)
    lift = estimate.difference
    margin_diff = estimate.margin_difference
    redemption_rate = float("nan")
    if study is not None:
        redemption_rate = study.redemption_summary()["redeemed_rate"]
    realized_cost = request.coupon_cents * redemption_rate
    return {
        "historical_estimate_label": estimate.label,
        "historical_lift_pp": lift * 100,
        "historical_lift_ci_pp": [estimate.ci_low * 100, estimate.ci_high * 100],
        "historical_lift_p_value": estimate.p_value,
        "historical_redemption_rate": redemption_rate,
        "historical_batch_mean_margin_cents": (
            estimate.treated_margin if np.isfinite(estimate.treated_margin) else None
        ),
        "coupon_units": n,
        "holdout_units": len(allocations) - n,
        "expected_incremental_repurchases": n * lift if n else 0.0,
        "expected_incremental_repurchases_range": [
            n * estimate.ci_low if n else 0.0,
            n * estimate.ci_high if n else 0.0,
        ],
        "expected_incremental_margin_cents": n * margin_diff if n else 0.0,
        "expected_incremental_margin_range_cents": [
            n * estimate.margin_ci_low if n else 0.0,
            n * estimate.margin_ci_high if n else 0.0,
        ],
        "reserved_coupon_cost_cents": n * request.coupon_cents,
        "realized_coupon_cost_cents": n * realized_cost if np.isfinite(redemption_rate) else None,
        "net_reserved_cents": n * (margin_diff - request.coupon_cents) if n else 0.0,
        "net_reserved_range_cents": [
            n * (estimate.margin_ci_low - request.coupon_cents) if n else 0.0,
            n * (estimate.margin_ci_high - request.coupon_cents) if n else 0.0,
        ],
        "net_realized_cents": (
            n * (margin_diff - realized_cost) if n and np.isfinite(redemption_rate) else None
        ),
        "demonstrated_positive_effect": bool(estimate.ci_low > 0),
        "extrapolation_boundary": (
            "历史效应来自随机留量批次中在发券前已有完成购买、且距上次购买 11 天的单位；"
            "不能外推到长期未购、从未购买或未登录访客，也不能把第 30 天后的复购算进来。"
        ),
    }


# ---------------------------------------------------------------------------
# Late binding reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LateBindingFinding:
    household: str
    binding_known_at: datetime
    coupons: tuple[dict, ...]
    redeemed_kept: tuple[str, ...]
    unsent_contacts_kept: tuple[str, ...]
    unsent_contacts_dropped: tuple[str, ...]
    reserved_cents_consumed: int
    contact_slots_consumed: int
    comparison_units_removed: tuple[str, ...]
    comparison_scope_before: int
    comparison_scope_after: int
    note: str


def reconcile_late_bindings(
    dataset: Dataset, *, as_of: datetime | None = None, start: datetime | None = None
) -> dict[str, object]:
    """Find bindings that merge coupon recipients *after* issuance.

    Redeemed benefits stay. The household stops serving as a coupon/no-coupon
    comparison. Among contacts that were never sent, only one entitled contact
    is kept. The budget and capacity already occupied by the duplicate are
    reported so operations can see what was squeezed out.
    """
    as_of = as_of or dataset.latest_checkpoint
    identity = IdentityMap(dataset, as_of=as_of)
    coupons_by_household: dict[str, list] = defaultdict(list)
    for coupon in dataset.coupons:
        coupons_by_household[identity.household_key(coupon.sent_to)].append(coupon)

    findings: list[LateBindingFinding] = []
    for household, coupons in sorted(coupons_by_household.items()):
        if len(coupons) < 2:
            continue
        involved = {c.sent_to for c in coupons}
        bindings = [
            b
            for b in dataset.bindings
            if b.verified and (b.left_handle in involved or b.right_handle in involved)
        ]
        if not bindings:
            continue
        binding_known_at = min(b.known_at for b in bindings)
        late = any(c.issued_at < binding_known_at for c in coupons)
        if not late:
            continue
        redeemed = tuple(c.coupon for c in coupons if c.state == "已兑")
        units = sorted({identity.unit_key(c.sent_to) for c in coupons})
        contactable = [unit for unit in units if identity.contact_allowed(unit)]
        # Going forward the household contributes one contact, chosen from the
        # units that still have contact permission. A unit that already
        # redeemed keeps its benefit. If nobody in the household can be
        # contacted, there is no "kept" unit at all; dropping everyone would
        # wrongly imply the redeemed benefit was withdrawn.
        kept_unit = contactable[0] if contactable else None
        kept = (kept_unit,) if kept_unit else ()
        dropped = tuple(unit for unit in units if unit not in kept)
        reserved = sum(c.cents for c in coupons[1:])
        mixed_offers = _mixed_arm_offers(dataset, identity, household)
        comparison_units = tuple(sorted(o.offer for o in mixed_offers))
        findings.append(
            LateBindingFinding(
                household=household,
                binding_known_at=binding_known_at,
                coupons=tuple(
                    {
                        "coupon": c.coupon,
                        "offer": c.offer,
                        "sent_to": c.sent_to,
                        "state": c.state,
                        "redeemed_by": c.redeemed_by,
                        "issued_at": c.issued_at.isoformat(),
                    }
                    for c in coupons
                ),
                redeemed_kept=redeemed,
                unsent_contacts_kept=kept,
                unsent_contacts_dropped=dropped,
                reserved_cents_consumed=reserved,
                contact_slots_consumed=len(dropped),
                comparison_units_removed=comparison_units,
                comparison_scope_before=len(
                    [o for o in dataset.offers if o.randomized]
                ),
                comparison_scope_after=len(
                    [o for o in dataset.offers if o.randomized]
                )
                - len(comparison_units),
                note=(
                    "已兑权益不撤销；该户不再用作发券/不发券对照；后续联系只保留有权的一次；"
                    "重复券继续占用预算与客服时段。"
                ),
            )
        )
    randomized_total = sum(1 for o in dataset.offers if o.randomized)
    removed = sum(len(f.comparison_units_removed) for f in findings)
    late_bound = 0
    for offer in dataset.offers:
        if not offer.randomized:
            continue
        involved = {
            b.known_at
            for b in dataset.bindings
            if b.verified and (b.left_handle == offer.handle or b.right_handle == offer.handle)
        }
        if involved and min(involved) > offer.assigned_at:
            late_bound += 1
    return {
        "findings": findings,
        "households_affected": len(findings),
        "comparison_units_removed": removed,
        "randomized_offers_total": randomized_total,
        "randomized_offers_remaining": randomized_total - removed,
        "late_bound_randomized_offers": late_bound,
        "reserved_cents_consumed": sum(f.reserved_cents_consumed for f in findings),
        "households_with_two_coupons": len(
            [1 for coupons in coupons_by_household.values() if len(coupons) > 1]
        ),
        "duplicate_coupons": sum(
            len(coupons) - 1 for coupons in coupons_by_household.values() if len(coupons) > 1
        ),
        "note": "晚绑定只影响此后判断，不改写绑定前已完成的分配。",
    }


def _mixed_arm_offers(dataset: Dataset, identity: IdentityMap, household: str) -> tuple:
    """Randomized offers in a household that contains both arms.

    Only this case loses comparison status. A household with two coupons under
    the *same* assignment is a doubled dose, not a broken contrast; it is
    reported in the estimate sensitivity instead of being deleted here.
    """
    offers = [
        offer
        for offer in dataset.offers
        if offer.randomized
        and identity.household_of_unit(identity.unit_key(offer.handle)) == household
    ]
    arms = {offer.assignment for offer in offers}
    return tuple(offers) if len(arms) > 1 else ()


# ---------------------------------------------------------------------------
# Campaign ledger and forecast reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    """What was promised at plan time, frozen as a file on disk."""

    campaign: str
    made_at: datetime
    eligibility_as_of: datetime
    budget_cents: int
    coupon_cents: int
    valid_until: datetime
    contact_slots: int
    coupon_units: tuple[str, ...]
    holdout_units: tuple[str, ...]
    expected_incremental_repurchases: float
    expected_net_cents: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    **asdict(self),
                    "made_at": self.made_at.isoformat(),
                    "eligibility_as_of": self.eligibility_as_of.isoformat(),
                    "valid_until": self.valid_until.isoformat(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ForecastRecord":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["made_at"] = datetime.fromisoformat(raw["made_at"])
        raw["eligibility_as_of"] = datetime.fromisoformat(raw["eligibility_as_of"])
        raw["valid_until"] = datetime.fromisoformat(raw["valid_until"])
        for key in ("coupon_units", "holdout_units"):
            raw[key] = tuple(raw[key])
        return cls(**raw)


def reconcile_forecast(
    record: ForecastRecord, dataset: Dataset, *, as_of: datetime | None = None
) -> dict[str, object]:
    """Compare a frozen forecast with what actually matured.

    The stored forecast is never rewritten. The function only reports actual
    redemptions and mature repurchases, so a later model cannot be used to
    claim the old campaign "predicted correctly".
    """
    as_of = as_of or dataset.latest_checkpoint
    identity = IdentityMap(dataset, as_of=as_of)
    book = OrderBook(dataset, identity, dataset.policy.repurchase_window_days)
    coupon_units = set(record.coupon_units)
    holdout_units = set(record.holdout_units)

    def outcomes(units: Iterable[str]) -> tuple[int, int, int, int]:
        repurchases = 0
        mature = 0
        margin = 0
        pending = 0
        for unit in units:
            last = book.last_purchase(unit, as_of=record.eligibility_as_of)
            if last is None:
                continue
            end = record.valid_until
            events = [
                e
                for e in book.by_unit.get(unit, [])
                if record.eligibility_as_of < e.paid_at <= end
                and e.purchase.returns_known_at <= as_of
                and book.is_active(e)
            ]
            waiting = any(
                record.eligibility_as_of < e.paid_at <= end
                and e.purchase.paid_at <= as_of
                and e.purchase.returns_known_at > as_of
                and e.purchase.replacement_of is None
                for e in book.by_unit.get(unit, [])
            )
            if end > as_of or waiting:
                pending += 1
                continue
            mature += 1
            repurchases += bool(events)
            margin += sum(e.margin for e in events)
        return repurchases, mature, margin, pending

    coupon_repurchases, coupon_mature, coupon_margin, coupon_pending = outcomes(coupon_units)
    holdout_repurchases, holdout_mature, holdout_margin, holdout_pending = outcomes(holdout_units)
    redeemed = sum(
        1
        for c in dataset.coupons
        if c.state == "已兑" and identity.unit_key(c.sent_to) in coupon_units
    )
    actual_net = coupon_margin - holdout_margin - redeemed * record.coupon_cents
    return {
        "campaign": record.campaign,
        "forecast_made_at": record.made_at.isoformat(),
        "eligibility_as_of": record.eligibility_as_of.isoformat(),
        "actual_coupon_units": len(coupon_units),
        "actual_holdout_units": len(holdout_units),
        "actual_redeemed": redeemed,
        "actual_coupon_repurchases": coupon_repurchases,
        "actual_coupon_mature": coupon_mature,
        "actual_coupon_pending": coupon_pending,
        "actual_holdout_repurchases": holdout_repurchases,
        "actual_holdout_mature": holdout_mature,
        "actual_holdout_pending": holdout_pending,
        "actual_incremental_repurchases": coupon_repurchases - holdout_repurchases,
        "forecast_incremental_repurchases": record.expected_incremental_repurchases,
        "forecast_net_cents": record.expected_net_cents,
        "mature": coupon_mature == len(coupon_units) and holdout_mature == len(holdout_units),
        "actual_net_cents": None if not (coupon_mature + holdout_mature) else actual_net,
        "actual_net_note": (
            "未成熟或仍有待定单位时不给实际净收益，避免把未兑券当成券成本或把未到期当成失败。"
        ),
        "note": (
            "冻结的预测不改写；未成熟时只报告状态，不把未兑或未到期计成失败。"
            "实际增量只按已成熟单位计算，待定单位既不计入成功也不计入失败。"
        ),
    }
