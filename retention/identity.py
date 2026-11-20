"""Identity resolution: payer units and households, both as-of a moment.

Two different notions of "the same customer" appear in the data and must not be
merged:

* ``confirmed_payer`` links account handles that are provably the same payer.
  An App handle and a storefront handle can share a confirmed payer; that is a
  duplicate account, not a new customer. Blank ``confirmed_payer`` is *not*
  evidence of a new payer.
* ``bindings`` with ``evidence == "核实户成员"`` link handles into a household.
  Households share coupons and people; they are not independent observations
  for offer experiments. ``仅同址`` is explicitly *not* a binding and never
  merges units.

``as_of`` trims both relations to what was known at a business moment, which is
what the offer analysis must use: a binding discovered after an assignment must
not be used to reclassify that assignment, but it does invalidate the pair
going forward.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from .data import Account, Binding, Dataset, DataError


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            # Deterministic orientation keeps labels stable across runs.
            if a <= b:
                self.parent[b] = a
            else:
                self.parent[a] = b


@dataclass(frozen=True, slots=True)
class Unit:
    key: str
    handles: tuple[str, ...]
    confirmed_payer: str | None
    household: str
    annuity: bool = False


class IdentityMap:
    """Resolve handles to payer units and households, optionally as-of a time.

    ``as_of=None`` uses everything on file (the present). Passing a moment only
    uses accounts recorded and bindings *known* by then.
    """

    def __init__(self, dataset: Dataset, *, as_of: datetime | None = None, require_confirmed: bool = False):
        self.dataset = dataset
        self.as_of = as_of
        self.require_confirmed = require_confirmed
        self._units = _UnionFind()
        self._households = _UnionFind()
        self._build()

    # -- construction ----------------------------------------------------
    def _account_visible(self, account: Account) -> bool:
        return self.as_of is None or account.recorded_at <= self.as_of

    def _binding_visible(self, binding: Binding) -> bool:
        if not binding.verified:
            return False
        return self.as_of is None or binding.known_at <= self.as_of

    def _build(self) -> None:
        accounts = [a for a in self.dataset.accounts if self._account_visible(a)]
        # Every visible handle is at least its own unit so lookups never fail.
        for account in self.dataset.accounts:
            self._units.find(account.handle)
            self._households.find(account.handle)
        by_payer: dict[str, list[str]] = defaultdict(list)
        for account in accounts:
            if account.confirmed_payer:
                by_payer[account.confirmed_payer].append(account.handle)
        for handles in by_payer.values():
            for handle in handles[1:]:
                self._units.union(handles[0], handle)
            if not self.require_confirmed:
                for handle in handles[1:]:
                    self._households.union(handles[0], handle)
        for binding in self.dataset.bindings:
            if self._binding_visible(binding):
                self._households.union(binding.left_handle, binding.right_handle)
        self._unit_members: dict[str, list[str]] = defaultdict(list)
        for account in self.dataset.accounts:
            self._unit_members[self.unit_key(account.handle)].append(account.handle)
        self._household_members: dict[str, list[str]] = defaultdict(list)
        for account in self.dataset.accounts:
            self._household_members[self.household_key(account.handle)].append(account.handle)

    # -- lookups ---------------------------------------------------------
    def is_visible(self, handle: str) -> bool:
        account = self.dataset.accounts_by_handle.get(handle)
        return account is not None and self._account_visible(account)

    def unit_key(self, handle: str) -> str:
        """Stable payer-unit key.

        Handles without a confirmed payer stay separate *and* are marked so
        downstream code can report them as an unconfirmed range rather than
        silently treating each as a new customer.
        """
        account = self.dataset.accounts_by_handle.get(handle)
        if account is None:
            raise DataError(f"unknown handle {handle}")
        if account.confirmed_payer:
            return "payer:" + account.confirmed_payer
        return "handle:" + handle

    def household_key(self, handle: str) -> str:
        account = self.dataset.accounts_by_handle.get(handle)
        if account is None:
            raise DataError(f"unknown handle {handle}")
        return "hh:" + self._households.find(handle)

    def handles_of_unit(self, unit_key: str) -> tuple[str, ...]:
        return tuple(sorted(self._unit_members.get(unit_key, ())))

    def handles_of_household(self, household_key: str) -> tuple[str, ...]:
        return tuple(sorted(self._household_members.get(household_key, ())))

    def household_of_unit(self, unit_key: str) -> str:
        handles = self.handles_of_unit(unit_key)
        if not handles:
            raise DataError(f"unit {unit_key} has no visible handles")
        roots = {self.household_key(handle) for handle in handles}
        # A unit whose handles are bound into one household has a single root.
        if len(roots) > 1:
            return "hh:" + min(roots)
        return roots.pop()

    def unit_is_confirmed(self, unit_key: str) -> bool:
        return unit_key.startswith("payer:")

    def contact_allowed(self, unit_key: str, *, requires_all: bool = True) -> bool:
        """Whether the unit may be contacted as of the map's moment.

        A unit may contain several handles. If any handle prohibits contact the
        unit is not contactable by default; callers can ask for "any handle"
        semantics explicitly.
        """
        handles = self.handles_of_unit(unit_key)
        if not handles:
            return False
        flags = []
        for handle in handles:
            account = self.dataset.accounts_by_handle[handle]
            if self.as_of is not None and account.recorded_at > self.as_of:
                continue
            flags.append(account.contact_permission == "允许联系")
        if not flags:
            return False
        return all(flags) if requires_all else any(flags)

    def unit_channels(self, unit_key: str) -> set[str]:
        out = set()
        for handle in self.handles_of_unit(unit_key):
            account = self.dataset.accounts_by_handle[handle]
            if self.as_of is None or account.recorded_at <= self.as_of:
                out.add(account.channel)
        return out


def duplicate_coupon_households(dataset: Dataset, *, as_of: datetime | None = None) -> dict[str, list[str]]:
    """Households holding more than one issued coupon, grouped by household."""
    identity = IdentityMap(dataset, as_of=as_of)
    grouped: dict[str, list[str]] = defaultdict(list)
    for coupon in dataset.coupons:
        grouped[identity.household_key(coupon.sent_to)].append(coupon.coupon)
    return {key: sorted(ids) for key, ids in grouped.items() if len(ids) > 1}
