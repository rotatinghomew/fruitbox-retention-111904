"""Loading and as-of validation of the source tables.

Source CSV files are treated as untrusted input: every field is parsed
explicitly, empty cells become ``None`` (never zero), referential integrity is
checked, and business timestamps are compared so that records that claim to
predate their own evidence fail loudly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

ISO = "%Y-%m-%dT%H:%M:%S%z"


class DataError(ValueError):
    """Raised when the source contract or business ordering is violated."""


def _dt(value: str | None, *, field_name: str) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.strptime(value, ISO)
    except ValueError as exc:  # pragma: no cover - defensive
        raise DataError(f"{field_name}: cannot parse timestamp {value!r}") from exc
    if parsed.tzinfo is None:  # pragma: no cover - defensive
        raise DataError(f"{field_name}: naive timestamp {value!r}")
    return parsed


def _text(value: str | None) -> str | None:
    return None if value is None or value == "" else value


def _int(value: str | None, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise DataError(f"{field_name}: not an integer: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class Account:
    handle: str
    confirmed_payer: str | None
    channel: str
    contact_permission: str
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class Binding:
    left_handle: str
    right_handle: str
    household: str | None
    known_at: datetime
    evidence: str
    effective_from: datetime

    @property
    def verified(self) -> bool:
        return self.evidence == "核实户成员"


@dataclass(frozen=True, slots=True)
class Purchase:
    order: str
    payer_handle: str
    recipient_handle: str | None
    source_context: str
    paid_at: datetime
    returns_known_at: datetime
    outcome: str
    replacement_of: str | None
    margin_cents: int | None


@dataclass(frozen=True, slots=True)
class Offer:
    offer: str
    handle: str
    batch: str
    mechanism: str
    assignment: str
    treat_probability: float | None
    assigned_at: datetime
    transfer_possible: bool

    @property
    def randomized(self) -> bool:
        return self.mechanism == "随机留量"


@dataclass(frozen=True, slots=True)
class Coupon:
    coupon: str
    offer: str
    sent_to: str
    cents: int
    issued_at: datetime
    expires_at: datetime
    state: str
    redeemed_by: str | None

    @property
    def redeemed(self) -> bool:
        return self.state == "已兑"


@dataclass(frozen=True, slots=True)
class CampaignPolicy:
    repurchase_window_days: int
    checkpoints: tuple[datetime, ...]
    budget_cents: int
    coupon_cents: int
    valid_until: datetime
    max_contacts_per_day: int
    preserve_holdout_share: float


@dataclass
class Dataset:
    accounts: list[Account]
    bindings: list[Binding]
    purchases: list[Purchase]
    offers: list[Offer]
    coupons: list[Coupon]
    policy: CampaignPolicy
    root: Path | None = None

    def __post_init__(self) -> None:
        self.accounts_by_handle = {a.handle: a for a in self.accounts}
        if len(self.accounts_by_handle) != len(self.accounts):
            raise DataError("accounts: duplicate handle")
        self.offers_by_id = {o.offer: o for o in self.offers}
        self.coupons_by_id = {c.coupon: c for c in self.coupons}
        self.purchases_by_order = {p.order: p for p in self.purchases}
        self.validate()

    # -- validation -----------------------------------------------------
    def validate(self) -> None:
        handles = set(self.accounts_by_handle)
        if len(self.offers_by_id) != len(self.offers):
            raise DataError("offers: duplicate offer id")
        if len(self.coupons_by_id) != len(self.coupons):
            raise DataError("coupons: duplicate coupon id")
        if len(self.purchases_by_order) != len(self.purchases):
            raise DataError("purchases: duplicate order id")
        for p in self.purchases:
            if p.payer_handle not in handles:
                raise DataError(f"{p.order}: unknown payer {p.payer_handle}")
            if p.recipient_handle is not None and p.recipient_handle not in handles:
                raise DataError(f"{p.order}: unknown recipient {p.recipient_handle}")
            if p.returns_known_at < p.paid_at:
                raise DataError(f"{p.order}: returns known before payment")
            if p.replacement_of is not None:
                original = self.purchases_by_order.get(p.replacement_of)
                if original is None:
                    raise DataError(f"{p.order}: unknown replacement_of")
                if original.paid_at > p.paid_at:
                    raise DataError(f"{p.order}: replacement precedes original")
                if p.outcome != "补发":
                    raise DataError(f"{p.order}: replacement row without 补发 outcome")
            if p.outcome == "全退" and (p.margin_cents or 0) != 0:
                raise DataError(f"{p.order}: full refund with non-zero margin")
        for o in self.offers:
            if o.handle not in handles:
                raise DataError(f"{o.offer}: unknown handle {o.handle}")
            if o.mechanism not in {"随机留量", "报名"}:
                raise DataError(f"{o.offer}: bad mechanism {o.mechanism}")
            if o.mechanism == "随机留量" and not (0 < (o.treat_probability or 0) <= 1):
                raise DataError(f"{o.offer}: randomized offer needs a probability")
            if o.mechanism != "随机留量" and o.treat_probability is not None:
                raise DataError(f"{o.offer}: self-selected offer must not carry a probability")
        for c in self.coupons:
            offer = self.offers_by_id.get(c.offer)
            if offer is None:
                raise DataError(f"{c.coupon}: unknown offer {c.offer}")
            if c.sent_to not in handles:
                raise DataError(f"{c.coupon}: unknown recipient {c.sent_to}")
            if c.redeemed_by is not None and c.redeemed_by not in handles:
                raise DataError(f"{c.coupon}: unknown redeemer {c.redeemed_by}")
            if c.issued_at < offer.assigned_at:
                raise DataError(f"{c.coupon}: issued before assignment")
            if c.expires_at <= c.issued_at:
                raise DataError(f"{c.coupon}: expires before issue")
            if c.sent_to != offer.handle:
                raise DataError(f"{c.coupon}: sent to a handle that was not offered")
            if offer.assignment != "给券":
                raise DataError(f"{c.coupon}: coupon for a no-coupon assignment")
            if c.state not in {"已发", "已兑"}:
                raise DataError(f"{c.coupon}: unknown coupon state {c.state}")
            if (c.redeemed_by is None) == (c.state == "已兑"):
                raise DataError(f"{c.coupon}: redemption state and redeemer disagree")
        for b in self.bindings:
            if b.left_handle not in handles or b.right_handle not in handles:
                raise DataError("bindings: unknown handle")
            if b.known_at < b.effective_from:
                raise DataError("bindings: known before it became effective")
        for a in self.accounts:
            if a.channel == "匿名来客" and a.confirmed_payer is not None:
                raise DataError(f"{a.handle}: anonymous visitor with a confirmed payer")

    # -- policy helpers -------------------------------------------------
    def checkpoint_at_or_before(self, moment: datetime) -> datetime:
        candidates = [c for c in self.policy.checkpoints if c <= moment]
        if not candidates:
            raise DataError(f"no checkpoint at or before {moment.isoformat()}")
        return max(candidates)

    @property
    def latest_checkpoint(self) -> datetime:
        return max(self.policy.checkpoints)

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        accounts = [
            Account(
                handle=r["handle"],
                confirmed_payer=_text(r["confirmed_payer"]),
                channel=r["channel"],
                contact_permission=r["contact_permission"],
                recorded_at=_dt(r["recorded_at"], field_name="recorded_at"),
            )
            for r in _read(root / "accounts.csv")
        ]
        bindings = [
            Binding(
                left_handle=r["left_handle"],
                right_handle=r["right_handle"],
                household=_text(r["household"]),
                known_at=_dt(r["known_at"], field_name="known_at"),
                evidence=r["evidence"],
                effective_from=_dt(r["effective_from"], field_name="effective_from"),
            )
            for r in _read(root / "bindings.csv")
        ]
        purchases = [
            Purchase(
                order=r["order"],
                payer_handle=r["payer_handle"],
                recipient_handle=_text(r["recipient_handle"]),
                source_context=r["source_context"],
                paid_at=_dt(r["paid_at"], field_name="paid_at"),
                returns_known_at=_dt(r["returns_known_at"], field_name="returns_known_at"),
                outcome=r["outcome"],
                replacement_of=_text(r["replacement_of"]),
                margin_cents=_int(r["margin_cents"], field_name="margin_cents"),
            )
            for r in _read(root / "purchases.csv")
        ]
        offers = [
            Offer(
                offer=r["offer"],
                handle=r["handle"],
                batch=r["batch"],
                mechanism=r["mechanism"],
                assignment=r["assignment"],
                treat_probability=(float(r["treat_probability"]) if r["treat_probability"] else None),
                assigned_at=_dt(r["assigned_at"], field_name="assigned_at"),
                transfer_possible=r["transfer_possible"] == "1",
            )
            for r in _read(root / "offers.csv")
        ]
        coupons = [
            Coupon(
                coupon=r["coupon"],
                offer=r["offer"],
                sent_to=r["sent_to"],
                cents=_int(r["cents"], field_name="cents"),
                issued_at=_dt(r["issued_at"], field_name="issued_at"),
                expires_at=_dt(r["expires_at"], field_name="expires_at"),
                state=r["state"],
                redeemed_by=_text(r["redeemed_by"]),
            )
            for r in _read(root / "coupons.csv")
        ]
        raw_policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))
        current = raw_policy["current_campaign"]
        policy = CampaignPolicy(
            repurchase_window_days=int(raw_policy["repurchase_window_days"]),
            checkpoints=tuple(_dt(c, field_name="checkpoint") for c in raw_policy["checkpoints"]),
            budget_cents=int(current["budget_cents"]),
            coupon_cents=int(current["coupon_cents"]),
            valid_until=_dt(current["valid_until"], field_name="valid_until"),
            max_contacts_per_day=int(current["max_contacts_per_day"]),
            preserve_holdout_share=float(current["preserve_holdout_share"]),
        )
        return cls(
            accounts=accounts,
            bindings=bindings,
            purchases=purchases,
            offers=offers,
            coupons=coupons,
            policy=policy,
            root=root,
        )


def _read(path: Path) -> Iterable[dict[str, str | None]]:
    if not path.exists():
        raise DataError(f"missing source file: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if None in row or None in row.values():
                raise DataError(f"{path.name}: ragged row")
            yield {key: _text(value) for key, value in row.items()}
