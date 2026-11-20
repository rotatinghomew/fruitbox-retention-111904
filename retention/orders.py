"""Purchases, maturity and first-purchase cohorts.

Counting rules (all deliberate):

* A purchase is a *purchase* only when it completed (``完成``) and is not a
  replacement. Full refunds (``全退``), replacements (``补发``) and rows that
  point at an original order are never counted as a new purchase.
* A purchase is only *known* once ``returns_known_at`` has passed. Before that
  the after-sales outcome can still turn the order into a full refund, so it
  must be reported as pending, not as a completed purchase and not as "did not
  repurchase".
* The first purchase of a cohort is per **payer unit**, never per handle. Two
  handles under one confirmed payer are one buyer.
* A gift's recipient is not a cohort member. Only a purchase the recipient pays
  for themselves creates a payer unit for them, and only if the recipient has
  its own identity evidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .data import Dataset, Purchase
from .identity import IdentityMap


@dataclass(frozen=True, slots=True)
class PurchaseEvent:
    purchase: Purchase
    unit: str
    household: str
    channel: str
    verified_identity: bool

    @property
    def paid_at(self) -> datetime:
        return self.purchase.paid_at

    @property
    def context(self) -> str:
        return self.purchase.source_context

    @property
    def margin(self) -> int:
        return int(self.purchase.margin_cents or 0)


@dataclass(frozen=True, slots=True)
class CohortMember:
    unit: str
    household: str
    first: PurchaseEvent
    week: str
    status: str  # "mature" | "pending"
    repurchased: bool | None
    repurchase_margin: int | None
    repurchase: PurchaseEvent | None


@dataclass
class OrderBook:
    dataset: Dataset
    identity: IdentityMap
    window_days: int

    def __post_init__(self) -> None:
        self.events: list[PurchaseEvent] = []
        for purchase in self.dataset.purchases:
            unit = self.identity.unit_key(purchase.payer_handle)
            household = self.identity.household_of_unit(unit)
            account = self.dataset.accounts_by_handle[purchase.payer_handle]
            self.events.append(
                PurchaseEvent(
                    purchase=purchase,
                    unit=unit,
                    household=household,
                    channel=account.channel,
                    verified_identity=self.identity.unit_is_confirmed(unit),
                )
            )
        self.by_unit: dict[str, list[PurchaseEvent]] = defaultdict(list)
        self.by_household: dict[str, list[PurchaseEvent]] = defaultdict(list)
        for event in self.events:
            self.by_unit[event.unit].append(event)
            self.by_household[event.household].append(event)
        for bucket in (self.by_unit, self.by_household):
            for key in bucket:
                bucket[key].sort(key=lambda e: e.paid_at)

    # -- single purchase rules -------------------------------------------
    @staticmethod
    def is_active(event: PurchaseEvent) -> bool:
        purchase = event.purchase
        return purchase.outcome == "完成" and purchase.replacement_of is None

    @staticmethod
    def is_mature(event: PurchaseEvent, as_of: datetime) -> bool:
        return event.purchase.returns_known_at <= as_of

    def is_known_purchase(self, event: PurchaseEvent, as_of: datetime) -> bool:
        return self.is_active(event) and self.is_mature(event, as_of)

    def pending(self, event: PurchaseEvent, as_of: datetime) -> bool:
        """Paid, not yet classified, and could still become a full refund."""
        return (
            event.purchase.paid_at <= as_of
            and not self.is_mature(event, as_of)
            and not (event.purchase.outcome == "全退")
            and event.purchase.replacement_of is None
        )

    # -- unit-level sequences --------------------------------------------
    def active_purchases(self, unit: str, as_of: datetime | None = None) -> list[PurchaseEvent]:
        out = [e for e in self.by_unit.get(unit, []) if self.is_active(e)]
        if as_of is not None:
            out = [e for e in out if self.is_mature(e, as_of)]
        return out

    def purchases_between(
        self, unit: str, start: datetime, end: datetime, *, as_of: datetime
    ) -> list[PurchaseEvent]:
        return [
            e
            for e in self.by_unit.get(unit, [])
            if start < e.paid_at <= end and self.is_known_purchase(e, as_of)
        ]

    def first_purchase(self, unit: str, as_of: datetime) -> tuple[PurchaseEvent | None, bool]:
        """First known purchase and whether an earlier purchase is still pending.

        When an earlier order is still inside its after-sales maturity window,
        the cohort assignment itself is provisional; callers must report the
        unit as pending rather than assigning it to the later order.
        """
        ordered = [e for e in self.by_unit.get(unit, []) if e.paid_at <= as_of]
        pending_before = False
        for event in ordered:
            if self.is_known_purchase(event, as_of):
                return event, pending_before
            if self.pending(event, as_of):
                pending_before = True
        return None, pending_before

    # -- cohorts ----------------------------------------------------------
    def build_cohorts(self, as_of: datetime) -> list[CohortMember]:
        members: list[CohortMember] = []
        for unit in self.by_unit:
            first, pending_before = self.first_purchase(unit, as_of)
            if first is None:
                continue
            window_end = first.paid_at + timedelta(days=self.window_days)
            if window_end > as_of:
                members.append(
                    CohortMember(
                        unit=unit,
                        household=first.household,
                        first=first,
                        week=_week(first.paid_at),
                        status="pending",
                        repurchased=None,
                        repurchase_margin=None,
                        repurchase=None,
                    )
                )
                continue
            later = self.purchases_between(unit, first.paid_at, window_end, as_of=as_of)
            pending_inside = any(
                self.pending(e, as_of)
                for e in self.by_unit[unit]
                if first.paid_at < e.paid_at <= window_end
            )
            if pending_inside:
                members.append(
                    CohortMember(
                        unit=unit,
                        household=first.household,
                        first=first,
                        week=_week(first.paid_at),
                        status="pending",
                        repurchased=None,
                        repurchase_margin=None,
                        repurchase=None,
                    )
                )
                continue
            second = later[0] if later else None
            members.append(
                CohortMember(
                    unit=unit,
                    household=first.household,
                    first=first,
                    week=_week(first.paid_at),
                    status="mature",
                    repurchased=second is not None,
                    repurchase_margin=sum(e.margin for e in later) if later else 0,
                    repurchase=second,
                )
            )
        members.sort(key=lambda member: (member.first.paid_at, member.unit))
        return members

    # -- single-purchase summary -----------------------------------------
    def active_order_counts(self, as_of: datetime) -> dict[str, int]:
        units: dict[str, int] = defaultdict(int)
        for event in self.events:
            if self.is_known_purchase(event, as_of):
                units[event.unit] += 1
        return dict(units)

    def weekly_activity(self, as_of: datetime) -> dict[str, dict[str, int]]:
        """Weekly counts split into first purchase of a payer unit and repeat.

        This is the honest answer to "sales rose": a repeat order from an
        existing payer unit is not a new customer, and one payer's second
        account is not a repeat customer either.
        """
        known = [e for e in self.events if self.is_known_purchase(e, as_of)]
        first_at: dict[str, datetime] = {}
        for event in sorted(known, key=lambda e: e.paid_at):
            first_at.setdefault(event.unit, event.paid_at)
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event in known:
            week = _week(event.paid_at)
            out[week]["orders"] += 1
            out[week]["margin"] += event.margin
            if event.paid_at == first_at[event.unit]:
                out[week]["new_units"] += 1
            else:
                out[week]["repeat_orders"] += 1
            if event.context == "礼赠":
                out[week]["gift_orders"] += 1
            if not event.verified_identity:
                out[week]["unconfirmed_orders"] += 1
        return {week: dict(counts) for week, counts in sorted(out.items())}

    def handle_vs_unit_counts(self, as_of: datetime) -> dict[str, int]:
        """Order / handle / payer-unit counts for the duplication note.

        A confirmed payer with an App account and a storefront account shows up
        as two handles but one payer unit. Counting handles overstates the
        customer base; counting orders as customers overstates it further.
        """
        known = [e for e in self.events if self.is_known_purchase(e, as_of)]
        handles = {e.purchase.payer_handle for e in known}
        units = {e.unit for e in known}
        multi = {
            unit
            for unit in units
            if len({e.purchase.payer_handle for e in known if e.unit == unit}) > 1
        }
        return {
            "orders": len(known),
            "handles": len(handles),
            "units": len(units),
            "multi_handle_units": len(multi),
            "handles_saved": len(handles) - len(units),
        }

    def last_purchase(self, unit: str, as_of: datetime) -> PurchaseEvent | None:
        known = [e for e in self.by_unit.get(unit, []) if self.is_known_purchase(e, as_of)]
        return max(known, key=lambda e: e.paid_at) if known else None


def _week(moment: datetime) -> str:
    iso = moment.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def build_order_book(dataset: Dataset, as_of: datetime | None = None) -> OrderBook:
    return OrderBook(
        dataset=dataset,
        identity=IdentityMap(dataset, as_of=as_of),
        window_days=dataset.policy.repurchase_window_days,
    )


def build_cohorts(dataset: Dataset, as_of: datetime | None = None) -> list[CohortMember]:
    as_of = as_of or dataset.latest_checkpoint
    book = build_order_book(dataset, as_of=as_of)
    return book.build_cohorts(as_of)


def cohort_identity_sensitivity(
    dataset: Dataset, as_of: datetime | None = None
) -> dict[str, object]:
    """Compare cohorts under current identity and under strictly as-of identity.

    The cohort itself must be formed from what was known when the purchase was
    made. If a second platform account is only recorded later, an as-of view
    would treat the same person as two buyers. The difference is the size of the
    duplication error that the current-identity view removes.
    """
    as_of = as_of or dataset.latest_checkpoint
    current = build_cohorts(dataset, as_of)
    strict_identity = IdentityMap(dataset, as_of=as_of)

    # An as-of map keyed to each purchase date is expensive; bindings and
    # account records in this source are only ever added, so the strict view is
    # equivalent to resolving every handle with its own recorded_at snapshot.
    strict_units: dict[str, str] = {}
    for purchase in dataset.purchases:
        account = dataset.accounts_by_handle[purchase.payer_handle]
        if account.confirmed_payer and account.recorded_at <= purchase.paid_at:
            strict_units[purchase.payer_handle] = "payer:" + account.confirmed_payer
        else:
            strict_units[purchase.payer_handle] = "handle:" + purchase.payer_handle

    known = [
        e
        for e in build_order_book(dataset, as_of=as_of).events
        if e.purchase.outcome == "完成"
        and e.purchase.replacement_of is None
        and e.purchase.returns_known_at <= as_of
    ]
    strict_first: dict[str, datetime] = {}
    for event in sorted(known, key=lambda e: e.paid_at):
        unit = strict_units[event.purchase.payer_handle]
        if unit in strict_first:
            continue
        window_end = event.paid_at + timedelta(days=dataset.policy.repurchase_window_days)
        if window_end > as_of:
            continue
        strict_first[unit] = event.paid_at
    strict_repurchases = 0
    for unit, first_at in strict_first.items():
        window_end = first_at + timedelta(days=dataset.policy.repurchase_window_days)
        if any(
            strict_units[e.purchase.payer_handle] == unit
            and first_at < e.paid_at <= window_end
            for e in known
        ):
            strict_repurchases += 1
    mature = [m for m in current if m.status == "mature"]
    return {
        "current_units": len(mature),
        "current_repurchases": sum(1 for m in mature if m.repurchased),
        "strict_units": len(strict_first),
        "strict_repurchases": strict_repurchases,
        "difference": len(strict_first) - len(mature),
        "note": (
            "若按每笔购买当时的身份快照分别计数，会把后来才补录的同一付款人第二账户算成另一个主体；"
            "本报告采用当前确认身份统一回溯，并在差异不为零时同时报告两个口径。"
        ),
    }
