"""Historical coupon offers: what can and cannot be learned from them.

Design decisions that keep the estimate honest:

* Only ``随机留量`` offers enter the treatment/control contrast. ``报名`` units
  self-selected and have no counterfactual; they are reported separately.
* An offer is only evaluable once the whole 30-day window is mature at the
  evaluation checkpoint. The November batch is therefore pending, not a zero.
* An offer only tests *retention* if the unit had a completed purchase before
  the assignment. Offers made at first purchase are separated.
* A household that contains both a treated and a control unit is destroyed as
  a comparison: the control can use the treated member's coupon and the two
  cannot be assumed independent. The affected offers are excluded from the
  primary contrast and the scope is reported.
* ``transfer_possible`` is a pre-assignment flag concentrated in units with no
  post-offer purchase at all. It is kept as a separate stratum, never silently
  averaged into "the coupon effect".
* Redemption is an outcome, not the treatment. The per-redeemer effect is not
  identified by this design and is not reported as if it were.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from .data import Dataset, Offer
from .identity import IdentityMap
from .orders import OrderBook
from .stats import bootstrap_p_value, cluster_bootstrap, diff_in_means


@dataclass(frozen=True, slots=True)
class PrePeriod:
    purchases: int
    first_context: str | None
    last_context: str | None
    days_since_last: int | None
    mean_margin: float
    gift_share: float
    group_share: float
    channel: str


@dataclass(frozen=True, slots=True)
class OfferObservation:
    offer: Offer
    unit: str
    household: str
    treated: bool
    randomized: bool
    transfer: bool
    self_selected: bool
    prior_purchase: bool
    outcome: bool | None
    outcome_pending: bool
    future_margin: int | None
    pre: PrePeriod
    contaminated: bool
    duplicate_coupon_household: bool
    contact_violation: bool
    redemption: bool
    redeemed_by_other_handle: bool
    batch: str

    @property
    def arm(self) -> str:
        return "treated" if self.treated else "control"


@dataclass(frozen=True, slots=True)
class Estimate:
    label: str
    treated_n: int
    control_n: int
    treated_rate: float
    control_rate: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    treated_margin: float
    control_margin: float
    margin_difference: float
    margin_ci_low: float
    margin_ci_high: float
    notes: tuple[str, ...] = ()

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


@dataclass(frozen=True, slots=True)
class BatchLedger:
    batch: str
    assigned: int
    treated_offers: int
    control_offers: int
    self_selected: int
    coupons_issued: int
    coupons_redeemed: int
    coupon_budget_cents: int
    prohibited_contact_coupons: int
    duplicate_household_coupons: int
    mature: bool
    window_end: datetime


class OfferStudy:
    def __init__(self, dataset: Dataset, *, as_of: datetime | None = None):
        self.dataset = dataset
        self.as_of = as_of or dataset.latest_checkpoint
        self.identity = IdentityMap(dataset, as_of=self.as_of)
        self.book = OrderBook(dataset, self.identity, dataset.policy.repurchase_window_days)
        self.window = timedelta(days=dataset.policy.repurchase_window_days)
        self._build()

    # -- construction ----------------------------------------------------
    def _pre_period(self, unit: str, moment: datetime, price: str) -> PrePeriod:
        prior = [
            e
            for e in self.book.by_unit.get(unit, [])
            if e.paid_at < moment and e.purchase.returns_known_at <= moment and self.book.is_active(e)
        ]
        if prior:
            first = min(prior, key=lambda e: e.paid_at)
            last = max(prior, key=lambda e: e.paid_at)
            days = (moment - last.paid_at).days
            first_context = first.context
            last_context = last.context
        else:
            first_context = last_context = None
            days = None
        margins = [e.margin for e in prior]
        total = sum(margins)
        return PrePeriod(
            purchases=len(prior),
            first_context=first_context,
            last_context=last_context,
            days_since_last=days,
            mean_margin=float(np.mean(margins)) if margins else 0.0,
            gift_share=(sum(1 for e in prior if e.context == "礼赠") / len(prior)) if prior else 0.0,
            group_share=(sum(1 for e in prior if e.context == "团购") / len(prior)) if prior else 0.0,
            channel=price,
        )

    def _build(self) -> None:
        self.coupons_by_offer: dict[str, list] = defaultdict(list)
        for coupon in self.dataset.coupons:
            self.coupons_by_offer[coupon.offer].append(coupon)
        # Households that ever contain both arms among randomized offers.
        by_household: dict[str, list[Offer]] = defaultdict(list)
        for offer in self.dataset.offers:
            if offer.randomized:
                by_household[self.identity.household_of_unit(self.identity.unit_key(offer.handle))].append(offer)
        self.contaminated_households = {
            key
            for key, offers in by_household.items()
            if len({o.assignment for o in offers}) > 1
        }
        # A household that received more than one coupon has doubled exposure:
        # its treated unit is not a one-coupon observation. It is tracked
        # separately from the mixed-arm case.
        coupon_count: dict[str, int] = defaultdict(int)
        for coupon in self.dataset.coupons:
            coupon_count[self.identity.household_key(coupon.sent_to)] += 1
        self.duplicate_households = {key for key, count in coupon_count.items() if count > 1}
        self._mixed_household_offers = {
            o.offer: key
            for key, offers in by_household.items()
            if key in self.contaminated_households
            for o in offers
        }
        self.observations: list[OfferObservation] = []
        for offer in self.dataset.offers:
            unit = self.identity.unit_key(offer.handle)
            household = self.identity.household_of_unit(unit)
            prior = [
                e
                for e in self.book.by_unit.get(unit, [])
                if e.paid_at < offer.assigned_at and e.purchase.returns_known_at <= offer.assigned_at and self.book.is_active(e)
            ]
            window_end = offer.assigned_at + self.window
            mature = window_end <= self.as_of
            future = [
                e
                for e in self.book.by_unit.get(unit, [])
                if offer.assigned_at < e.paid_at <= window_end
            ]
            future_active = [e for e in future if e.purchase.outcome == "完成" and e.purchase.replacement_of is None]
            pending = any(
                e.purchase.paid_at <= self.as_of
                and e.purchase.returns_known_at > self.as_of
                and e.purchase.outcome != "全退"
                and e.purchase.replacement_of is None
                for e in future
            )
            coupon = self.coupons_by_offer.get(offer.offer, [])
            redeemed = any(c.state == "已兑" for c in coupon)
            redeemed_other = any(
                c.state == "已兑" and c.redeemed_by and c.redeemed_by != c.sent_to for c in coupon
            )
            account = self.dataset.accounts_by_handle[offer.handle]
            contact_violation = account.contact_permission == "禁止联系"
            if not mature:
                outcome: bool | None = None
            elif pending:
                outcome = None
            else:
                outcome = bool(future_active)
            self.observations.append(
                OfferObservation(
                    offer=offer,
                    unit=unit,
                    household=household,
                    treated=offer.assignment == "给券",
                    randomized=offer.randomized,
                    transfer=offer.transfer_possible,
                    self_selected=offer.mechanism == "报名",
                    prior_purchase=bool(prior),
                    outcome=outcome,
                    outcome_pending=not mature or pending,
                    future_margin=sum(e.margin for e in future_active) if outcome is not None else None,
                    pre=self._pre_period(unit, offer.assigned_at, account.channel),
                    contaminated=household in self.contaminated_households,
                    duplicate_coupon_household=household in self.duplicate_households,
                    contact_violation=contact_violation,
                    redemption=redeemed,
                    redeemed_by_other_handle=redeemed_other,
                    batch=offer.batch,
                )
            )

    # -- filters ---------------------------------------------------------
    def randomized_observations(self) -> list[OfferObservation]:
        return [
            o
            for o in self.observations
            if o.randomized and o.prior_purchase and not o.outcome_pending and o.outcome is not None
        ]

    def self_selected_pending(self) -> list[OfferObservation]:
        return [
            o
            for o in self.observations
            if o.self_selected and (not o.prior_purchase or o.outcome_pending)
        ]

    def pending_observations(self) -> list[OfferObservation]:
        return [o for o in self.observations if o.randomized and o.prior_purchase and o.outcome_pending]

    def self_selected_observations(self) -> list[OfferObservation]:
        return [
            o
            for o in self.observations
            if o.self_selected and o.prior_purchase and not o.outcome_pending and o.outcome is not None
        ]

    # -- estimation ------------------------------------------------------
    def estimate(
        self,
        observations: Sequence[OfferObservation],
        *,
        label: str,
        exclude_contaminated: bool = True,
        exclude_transfer: bool = False,
        exclude_contact_violation: bool = False,
        exclude_duplicate_household: bool = False,
    ) -> Estimate:
        rows = [
            {
                "treated": o.treated,
                "outcome": 1.0 if o.outcome else 0.0,
                "margin": float(o.future_margin or 0),
                "household": o.household,
                "batch": o.batch,
                "transfer": o.transfer,
            }
            for o in observations
            if (not exclude_contaminated or not o.contaminated)
            and (not exclude_transfer or not o.transfer)
            and (not exclude_contact_violation or not o.contact_violation)
            and (not exclude_duplicate_household or not o.duplicate_coupon_household)
        ]
        notes: list[str] = []
        if exclude_contaminated:
            excluded = sum(1 for o in observations if o.contaminated)
            notes.append(f"排除户内混合分配的观察 {excluded} 条")
        if exclude_transfer:
            notes.append("排除可跨账户转让的券")
        if exclude_contact_violation:
            notes.append("排除发券时已禁止联系的对象")
        if exclude_duplicate_household:
            excluded_dupes = sum(1 for o in observations if o.duplicate_coupon_household)
            notes.append(f"排除同户收到两张券的观察 {excluded_dupes} 条")
        if (
            len(rows) < 6
            or not any(r["treated"] for r in rows)
            or not any(not r["treated"] for r in rows)
        ):
            return Estimate(
                label=label,
                treated_n=sum(1 for r in rows if r["treated"]),
                control_n=sum(1 for r in rows if not r["treated"]),
                treated_rate=float("nan"),
                control_rate=float("nan"),
                difference=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                p_value=float("nan"),
                treated_margin=float("nan"),
                control_margin=float("nan"),
                margin_difference=float("nan"),
                margin_ci_low=float("nan"),
                margin_ci_high=float("nan"),
                notes=tuple(notes + ["样本不足，未估计"]),
            )
        treated = [r["outcome"] for r in rows if r["treated"]]
        control = [r["outcome"] for r in rows if not r["treated"]]
        difference = float(np.mean(treated) - np.mean(control))
        low, high, values = cluster_bootstrap(
            rows, lambda sample: diff_in_means(sample, outcome="outcome"), cluster="household"
        )
        p_value = bootstrap_p_value(values)
        margin_low, margin_high, _ = cluster_bootstrap(
            rows, lambda sample: diff_in_means(sample, outcome="margin"), cluster="household"
        )
        return Estimate(
            label=label,
            treated_n=len(treated),
            control_n=len(control),
            treated_rate=float(np.mean(treated)),
            control_rate=float(np.mean(control)),
            difference=difference,
            ci_low=low,
            ci_high=high,
            p_value=p_value,
            treated_margin=float(np.mean([r["margin"] for r in rows if r["treated"]])),
            control_margin=float(np.mean([r["margin"] for r in rows if not r["treated"]])),
            margin_difference=diff_in_means(rows, outcome="margin"),
            margin_ci_low=margin_low,
            margin_ci_high=margin_high,
            notes=tuple(notes),
        )

    def primary(self) -> Estimate:
        return self.estimate(self.randomized_observations(), label="随机留量 ITT（主要）")

    def sensitivity(self) -> dict[str, Estimate]:
        randomized = self.randomized_observations()
        return {
            "包括受污染户": self.estimate(randomized, label="包括户内混合分配的观察", exclude_contaminated=False),
            "排除可转让券": self.estimate(randomized, label="排除可跨账户转让的券", exclude_transfer=True),
            "排除发券时禁止联系": self.estimate(
                randomized, label="排除发券时已禁止联系的对象", exclude_contact_violation=True
            ),
            "排除同户重复券": self.estimate(
                randomized, label="排除同户收到两张券的观察", exclude_duplicate_household=True
            ),
            "全部谨慎排除": self.estimate(
                randomized,
                label="同时排除受污染户、可转让券、禁止联系与同户重复券",
                exclude_transfer=True,
                exclude_contact_violation=True,
                exclude_duplicate_household=True,
            ),
        }

    def by_stratum(self) -> dict[str, dict[str, Estimate]]:
        randomized = self.randomized_observations()
        out: dict[str, dict[str, Estimate]] = {}
        for field_name in ("transfer", "batch", "channel", "first_context"):
            groups: dict[str, list[OfferObservation]] = defaultdict(list)
            for observation in randomized:
                if field_name == "channel":
                    key = observation.pre.channel
                elif field_name == "first_context":
                    key = observation.pre.first_context or "未知"
                else:
                    key = str(getattr(observation, field_name))
                groups[key].append(observation)
            out[field_name] = {
                key: self.estimate(group, label=f"{field_name}={key}", exclude_contaminated=False)
                for key, group in sorted(groups.items())
            }
        # Pooled most-recent batches: a single batch is too small to separate
        # signal from noise, and batch-level intervals overlap heavily.
        recent = [o for o in randomized if o.batch >= "E10" and not o.contaminated]
        earlier = [o for o in randomized if o.batch < "E10" and not o.contaminated]
        out["period"] = {
            "E00-E09": self.estimate(earlier, label="早期批次 E00-E09"),
            "E10-E14": self.estimate(recent, label="最近批次 E10-E14"),
        }
        return out

    def transfer_summary(self) -> dict[str, object]:
        randomized = self.randomized_observations()
        transfer = [o for o in randomized if o.transfer]
        fixed = [o for o in randomized if not o.transfer]
        return {
            "transfer_n": len(transfer),
            "transfer_treated": sum(1 for o in transfer if o.treated),
            "transfer_repurchases": sum(1 for o in transfer if o.outcome),
            "fixed_n": len(fixed),
            "fixed_estimate": self.estimate(fixed, label="不可转让券", exclude_contaminated=True),
        }

    def redemption_summary(self) -> dict[str, object]:
        randomized = [o for o in self.randomized_observations() if o.treated]
        redeemed = [o for o in randomized if o.redemption]
        unredeemed = [o for o in randomized if not o.redemption]
        return {
            "treated_n": len(randomized),
            "redeemed_n": len(redeemed),
            "redeemed_rate": len(redeemed) / len(randomized) if randomized else float("nan"),
            "redeemed_repurchase": sum(1 for o in redeemed if o.outcome),
            "redeemed_repurchase_rate": (
                sum(1 for o in redeemed if o.outcome) / len(redeemed) if redeemed else float("nan")
            ),
            "unredeemed_n": len(unredeemed),
            "unredeemed_repurchase": sum(1 for o in unredeemed if o.outcome),
            "unredeemed_repurchase_rate": (
                sum(1 for o in unredeemed if o.outcome) / len(unredeemed) if unredeemed else float("nan")
            ),
            "redeemed_by_other_handle": sum(1 for o in redeemed if o.redeemed_by_other_handle),
        }

    # -- budget and eligibility ledger -----------------------------------
    def ledger(self) -> list[BatchLedger]:
        by_batch: dict[str, list[Offer]] = defaultdict(list)
        for offer in self.dataset.offers:
            by_batch[offer.batch].append(offer)
        coupon_by_offer: dict[str, list] = defaultdict(list)
        for coupon in self.dataset.coupons:
            coupon_by_offer[coupon.offer].append(coupon)
        duplicate_coupons = {
            coupon.coupon
            for coupon in self.dataset.coupons
            if len(
                {
                    c.coupon
                    for c in self.dataset.coupons
                    if self.identity.household_key(c.sent_to)
                    == self.identity.household_key(coupon.sent_to)
                }
            )
            > 1
        }
        rows: list[BatchLedger] = []
        for batch, offers in sorted(by_batch.items()):
            coupons = [c for o in offers for c in coupon_by_offer.get(o.offer, [])]
            window_end = max(o.assigned_at + self.window for o in offers)
            rows.append(
                BatchLedger(
                    batch=batch,
                    assigned=len(offers),
                    treated_offers=sum(1 for o in offers if o.assignment == "给券"),
                    control_offers=sum(1 for o in offers if o.assignment == "不给券"),
                    self_selected=sum(1 for o in offers if o.mechanism == "报名"),
                    coupons_issued=len(coupons),
                    coupons_redeemed=sum(1 for c in coupons if c.state == "已兑"),
                    coupon_budget_cents=sum(c.cents for c in coupons),
                    prohibited_contact_coupons=sum(
                        1
                        for c in coupons
                        if self.dataset.accounts_by_handle[c.sent_to].contact_permission == "禁止联系"
                        and self.dataset.accounts_by_handle[c.sent_to].recorded_at <= c.issued_at
                    ),
                    duplicate_household_coupons=sum(1 for c in coupons if c.coupon in duplicate_coupons),
                    mature=window_end <= self.as_of,
                    window_end=window_end,
                )
            )
        return rows

    def net_benefit(self, estimate: Estimate, *, coupon_cents: int, redemption_rate: float) -> dict[str, float]:
        """Net contribution per treated unit under two cost conventions.

        ``reserved`` charges the full face value of every issued coupon, which
        is how the budget is committed. ``realized`` charges only the coupons
        that are actually redeemed, which is the expected P&L cost. Both are
        shown because the policy reserves face value while the accounts realise
        only redemptions.
        """
        break_even_reserved = coupon_cents / estimate.treated_margin if estimate.treated_margin else float("nan")
        break_even_realized = (
            coupon_cents * redemption_rate / estimate.treated_margin if estimate.treated_margin else float("nan")
        )
        return {
            "incremental_repurchase_per_treated": estimate.difference,
            "incremental_margin_per_treated": estimate.margin_difference,
            "reserved_cost_per_treated": coupon_cents,
            "realized_cost_per_treated": coupon_cents * redemption_rate,
            "net_reserved": estimate.margin_difference - coupon_cents,
            "net_realized": estimate.margin_difference - coupon_cents * redemption_rate,
            "net_reserved_low": estimate.margin_ci_low - coupon_cents,
            "net_reserved_high": estimate.margin_ci_high - coupon_cents,
            "net_realized_low": estimate.margin_ci_low - coupon_cents * redemption_rate,
            "net_realized_high": estimate.margin_ci_high - coupon_cents * redemption_rate,
            "break_even_lift_reserved": break_even_reserved,
            "break_even_lift_realized": break_even_realized,
            "margin_per_repurchase": estimate.treated_margin,
        }

    # -- temporal validation ---------------------------------------------
    def temporal_validation(self, *, cut: str = "E10") -> dict[str, object]:
        """Fit on batches before ``cut`` and evaluate on batches from ``cut``.

        Features use only information available before assignment. The label is
        the 30-day repurchase outcome. This is a response model, not an uplift
        model; it answers whether a pre-campaign score could have ranked
        customers at all. If it cannot, the report must not pretend that a
        newly fitted score would have picked the right customers.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        randomized = [o for o in self.randomized_observations() if not o.contaminated]
        train = [o for o in randomized if o.batch < cut]
        test = [o for o in randomized if o.batch >= cut]
        if len(train) < 20 or len(test) < 20:
            return {"available": False, "reason": "训练或验证批次样本不足"}
        contexts = ["自用", "礼赠", "团购"]

        def design(observations: Sequence[OfferObservation]) -> np.ndarray:
            rows = []
            for o in observations:
                rows.append(
                    [
                        o.pre.purchases,
                        o.pre.days_since_last or 0,
                        o.pre.mean_margin / 1000,
                        o.pre.gift_share,
                        o.pre.group_share,
                        float(o.pre.channel == "App"),
                        float(o.pre.first_context == "自用"),
                        float(o.pre.first_context == "礼赠"),
                        float(o.pre.first_context == "团购"),
                        float(o.treated),
                    ]
                )
            return np.asarray(rows, dtype=float)

        scaler = StandardScaler().fit(design(train))
        model = LogisticRegression(max_iter=2000, random_state=20261120).fit(
            scaler.transform(design(train)), [int(bool(o.outcome)) for o in train]
        )
        probabilities = model.predict_proba(scaler.transform(design(test)))[:, 1]
        labels = [int(bool(o.outcome)) for o in test]
        auc = float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else float("nan")

        # Pre-registered simple rule: first-purchase self-use vs gift/group, then recency.
        def rule_score(o: OfferObservation) -> float:
            base = {"自用": 2.0, "礼赠": 1.0, "团购": 1.0}.get(o.pre.first_context or "", 1.0)
            return base + (0.5 if (o.pre.days_since_last or 999) <= 60 else 0.0)

        ranked = sorted(test, key=rule_score, reverse=True)
        top_half = ranked[: max(1, len(ranked) // 2)]
        bottom_half = ranked[max(1, len(ranked) // 2):]
        rule_top = self.estimate(top_half, label="规则分上半", exclude_contaminated=False)
        rule_bottom = self.estimate(bottom_half, label="规则分下半", exclude_contaminated=False)
        return {
            "available": True,
            "cut": cut,
            "train_n": len(train),
            "test_n": len(test),
            "auc": auc,
            "probabilities": probabilities,
            "labels": labels,
            "rule_top": rule_top,
            "rule_bottom": rule_bottom,
            "model_coefficients": model.coef_[0].tolist(),
            "feature_names": [
                "pre_purchases",
                "days_since_last",
                "mean_margin_k",
                "gift_share",
                "group_share",
                "channel_app",
                "first_self",
                "first_gift",
                "first_group",
                "treated",
            ],
        }
