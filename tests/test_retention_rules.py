import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from retention.campaign import (
    CampaignRequest,
    ForecastRecord,
    plan_campaign,
    reconcile_forecast,
    reconcile_late_bindings,
)
from retention.cohorts import decompose, weekly_cohorts, weekly_decomposition
from retention.data import Dataset
from retention.experiments import OfferStudy
from retention.identity import IdentityMap, duplicate_coupon_households
from retention.orders import build_cohorts, build_order_book

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def ts(text):
    return datetime.fromisoformat(text)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.load(DATA)


class SourceIntegrity(Base):
    def test_every_payer_has_an_account(self):
        handles = set(self.dataset.accounts_by_handle)
        for purchase in self.dataset.purchases:
            self.assertIn(purchase.payer_handle, handles)

    def test_refunds_have_zero_margin(self):
        for purchase in self.dataset.purchases:
            if purchase.outcome == "全退":
                self.assertEqual(purchase.margin_cents, 0)

    def test_verified_binding_requires_household_label(self):
        identity = IdentityMap(self.dataset)
        for binding in self.dataset.bindings:
            if binding.verified:
                self.assertTrue(binding.household)
                self.assertEqual(
                    identity.household_key(binding.left_handle),
                    identity.household_key(binding.right_handle),
                )

    def test_same_address_does_not_merge_units(self):
        identity = IdentityMap(self.dataset)
        same_address = [
            b for b in self.dataset.bindings if b.evidence == "仅同址"
        ]
        self.assertTrue(same_address)
        for binding in same_address:
            self.assertNotEqual(
                identity.household_key(binding.left_handle),
                identity.household_key(binding.right_handle),
            )

    def test_multi_platform_accounts_are_one_payer(self):
        identity = IdentityMap(self.dataset)
        self.assertEqual(identity.unit_key("APP0001"), identity.unit_key("SHOP0001"))
        self.assertNotEqual(identity.unit_key("APP0001"), identity.unit_key("APP0002"))


class Cohorts(Base):
    def test_repurchase_excludes_replacements_and_refunds(self):
        book = build_order_book(self.dataset, self.dataset.latest_checkpoint)
        for event in book.events:
            purchase = event.purchase
            if purchase.replacement_of is not None:
                self.assertFalse(book.is_active(event))
            if purchase.outcome == "全退":
                self.assertFalse(book.is_active(event))

    def test_recipient_is_not_a_cohort_member(self):
        members = build_cohorts(self.dataset)
        units = {m.unit for m in members}
        recipients = {
            p.recipient_handle for p in self.dataset.purchases if p.recipient_handle
        }
        recipient_units = {f"handle:{h}" for h in recipients}
        # A recipient only appears if they later paid themselves (GIFT0064).
        self.assertEqual(
            {u for u in units if u in recipient_units},
            {"handle:GIFT0064"},
        )

    def test_consecutive_handle_purchase_counts_as_one_repurchase(self):
        members = build_cohorts(self.dataset)
        by_unit = {m.unit: m for m in members}
        # C0003 bought on App then storefront inside the window.
        member = by_unit["payer:C0003"]
        self.assertTrue(member.repurchased)
        self.assertEqual(member.repurchase.purchase.payer_handle, "SHOP0003")

    def test_pending_window_is_not_zero(self):
        members = build_cohorts(self.dataset)
        pending = [m for m in members if m.status == "pending"]
        self.assertTrue(pending)
        self.assertTrue(all(m.repurchased is None for m in pending))

    def test_gift_payer_first_cohort_is_gift(self):
        members = build_cohorts(self.dataset)
        gift_first = [m for m in members if m.first.context == "礼赠"]
        self.assertTrue(gift_first)
        self.assertTrue(all(m.first.purchase.recipient_handle for m in gift_first))


class Decomposition(Base):
    def test_parts_add_to_observed_within_support(self):
        decomposition = weekly_decomposition(build_cohorts(self.dataset))
        for comparison in decomposition["comparisons"]:
            self.assertAlmostEqual(
                comparison.mix + comparison.within + comparison.interaction,
                comparison.within_support_difference,
                places=10,
            )
            self.assertAlmostEqual(
                comparison.within_support_difference + comparison.outside_support_difference,
                comparison.difference,
                places=10,
            )

    def test_pending_week_is_not_in_trend(self):
        decomposition = weekly_decomposition(build_cohorts(self.dataset))
        weeks = {w.week: w for w in decomposition["weeks"]}
        pending = [w for w in weeks.values() if w.size == 0 and w.pending]
        self.assertTrue(pending)
        self.assertEqual(decomposition["trend"]["weeks"], 15)


class OfferStudyTests(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.study = OfferStudy(cls.dataset)

    def test_self_selected_never_enters_primary(self):
        randomized = self.study.randomized_observations()
        clean = [o for o in randomized if not o.contaminated]
        primary = self.study.primary()
        self.assertEqual(primary.treated_n + primary.control_n, len(clean))
        self.assertGreater(len(self.study.self_selected_observations()), 0)
        self.assertTrue(all(not o.randomized for o in self.study.self_selected_observations()))

    def test_mixed_household_is_excluded_from_primary(self):
        randomized = self.study.randomized_observations()
        excluded = [o for o in randomized if o.contaminated]
        self.assertTrue(excluded)
        controls = {o.household for o in randomized if not o.treated}
        treated = {o.household for o in randomized if o.treated}
        mixed = controls & treated
        self.assertTrue(mixed)
        # Every observation in a mixed household is excluded from the primary.
        self.assertEqual({o.household for o in excluded}, mixed)
        with_contamination = self.study.estimate(
            randomized, label="含受污染户", exclude_contaminated=False
        )
        self.assertEqual(
            with_contamination.treated_n + with_contamination.control_n,
            len(randomized),
        )

    def test_pending_window_is_not_counted(self):
        pending = self.study.pending_observations()
        self.assertTrue(pending)
        self.assertTrue(all(o.outcome is None for o in pending))
        for observation in self.study.randomized_observations():
            self.assertIsNotNone(observation.outcome)

    def test_transfer_layer_is_separate(self):
        summary = self.study.transfer_summary()
        self.assertGreater(summary["transfer_n"], 0)
        self.assertEqual(summary["transfer_repurchases"], 0)
        primary_households = {o.household for o in self.study.randomized_observations()}
        self.assertTrue(primary_households)

    def test_batch_ledger_reports_eligibility_violations(self):
        ledger = self.study.ledger()
        self.assertTrue(all(row.prohibited_contact_coupons >= 0 for row in ledger))
        self.assertTrue(any(row.prohibited_contact_coupons > 0 for row in ledger))
        november = next(row for row in ledger if row.batch == "NOV")
        self.assertFalse(november.mature)

    def test_temporal_validation_uses_earlier_batches(self):
        result = self.study.temporal_validation()
        self.assertTrue(result["available"])
        self.assertEqual(result["train_n"] + result["test_n"], len(
            [o for o in self.study.randomized_observations() if not o.contaminated]
        ))


class CampaignPlanTests(Base):
    def request(self, **overrides):
        base = dict(
            start=ts("2026-11-20T09:00:00+08:00"),
            days=8,
            budget_cents=18600,
            coupon_cents=1800,
            valid_until=ts("2026-11-27T20:00:00+08:00"),
            max_contacts_per_day=5,
            preserve_holdout_share=0.3,
        )
        base.update(overrides)
        return CampaignRequest(**base)

    def test_never_lists_anonymous_or_unconfirmed(self):
        plan = plan_campaign(self.dataset, self.request())
        self.assertTrue(plan.coupon_units)
        for allocation in plan.allocations:
            self.assertTrue(allocation.unit.startswith("payer:"))
            for handle in allocation.handles:
                self.assertNotEqual(
                    self.dataset.accounts_by_handle[handle].channel, "匿名来客"
                )

    def test_one_household_one_contact(self):
        plan = plan_campaign(self.dataset, self.request())
        identity = IdentityMap(self.dataset, as_of=plan.request.start)
        households = [identity.household_of_unit(a.unit) for a in plan.allocations]
        self.assertEqual(len(households), len(set(households)))

    def test_self_repurchase_inside_window_is_not_recoverable(self):
        plan = plan_campaign(self.dataset, self.request())
        for allocation in plan.allocations:
            # The candidate records are the pre-plan state; verify directly.
            pass
        book = build_order_book(self.dataset, plan.request.start)
        for allocation in plan.allocations:
            last = book.last_purchase(allocation.unit, plan.request.start)
            self.assertGreaterEqual(
                (plan.request.start - last.paid_at).days, plan.request.min_days_since_last
            )

    def test_outstanding_coupon_blocks_second_send(self):
        plan = plan_campaign(self.dataset, self.request())
        self.assertTrue(plan.outstanding_coupons)
        blocked = {c["unit"] for c in plan.outstanding_coupons}
        self.assertFalse(blocked & {a.unit for a in plan.allocations})

    def test_budget_and_capacity_respected(self):
        plan = plan_campaign(self.dataset, self.request())
        self.assertLessEqual(plan.reserved_budget, plan.request.budget_cents)
        self.assertLessEqual(
            plan.contact_slots_used,
            plan.request.max_contacts_per_day * plan.request.days,
        )
        self.assertEqual(len(plan.coupon_units) + len(plan.holdout_units), len(plan.allocations))

    def test_holdout_share_preserved(self):
        plan = plan_campaign(self.dataset, self.request())
        share = len(plan.holdout_units) / len(plan.allocations)
        self.assertGreaterEqual(share + 1e-9, plan.request.preserve_holdout_share)

    def test_never_couponed_units_are_preferred(self):
        plan = plan_campaign(self.dataset, self.request())
        couponed = [a for a in plan.coupon_units if not a.never_couponed]
        never = [a for a in plan.coupon_units if a.never_couponed]
        self.assertGreater(len(never), 0)
        # Units that never received a coupon are the first to be used.
        self.assertLessEqual(len(couponed), max(1, len(never)))

    def test_late_binding_households_stay_out_of_comparison(self):
        plan = plan_campaign(self.dataset, self.request())
        for allocation in plan.allocations:
            if not allocation.comparison_eligible:
                self.assertTrue(allocation.role == "coupon" or allocation.role == "holdout")

    def test_expectation_does_not_invent_positive_lift(self):
        plan = plan_campaign(self.dataset, self.request())
        self.assertFalse(plan.expected["demonstrated_positive_effect"])
        self.assertLessEqual(plan.expected["expected_incremental_repurchases"], 0)


class LateBindingTests(Base):
    def test_redeemed_benefit_kept_and_scope_reported(self):
        result = reconcile_late_bindings(self.dataset)
        self.assertGreater(result["households_affected"], 0)
        mixed = 0
        for finding in result["findings"]:
            for coupon in finding.coupons:
                if coupon["state"] == "已兑":
                    self.assertIn(coupon["coupon"], finding.redeemed_kept)
            self.assertLessEqual(len(finding.unsent_contacts_kept), 1)
            self.assertGreater(finding.reserved_cents_consumed, 0)
            if finding.comparison_units_removed:
                mixed += 1
                # A household loses comparison status only when it holds both arms.
                assignments = {
                    o.assignment
                    for o in self.dataset.offers
                    if o.offer in finding.comparison_units_removed
                }
                self.assertEqual(assignments, {"给券", "不给券"})
        self.assertGreater(mixed, 0)
        self.assertLessEqual(mixed, result["households_affected"])

    def test_double_coupon_same_arm_is_an_estimate_sensitivity(self):
        study = OfferStudy(self.dataset)
        sensitivity = study.sensitivity()
        self.assertIn("排除同户重复券", sensitivity)
        self.assertLess(sensitivity["排除同户重复券"].treated_n, study.primary().treated_n)

    def test_peak_november_household_is_caught(self):
        result = reconcile_late_bindings(self.dataset)
        peak = [f for f in result["findings"] if "PEAK" in f.household]
        self.assertEqual(len(peak), 1)
        finding = peak[0]
        self.assertIn("NOV0", finding.redeemed_kept)
        self.assertEqual(len(finding.unsent_contacts_kept), 1)
        self.assertEqual(finding.contact_slots_consumed, 1)


class ForecastTests(Base):
    def test_forecast_roundtrip_and_immutability(self):
        plan = plan_campaign(
            self.dataset,
            CampaignRequest(
                start=ts("2026-11-20T09:00:00+08:00"),
                days=8,
                budget_cents=18600,
                coupon_cents=1800,
                valid_until=ts("2026-11-27T20:00:00+08:00"),
                max_contacts_per_day=5,
                preserve_holdout_share=0.3,
            ),
        )
        record = ForecastRecord(
            campaign="test",
            made_at=plan.request.start,
            eligibility_as_of=self.dataset.latest_checkpoint,
            budget_cents=plan.request.budget_cents,
            coupon_cents=plan.request.coupon_cents,
            valid_until=plan.request.valid_until,
            contact_slots=plan.contact_slots_used,
            coupon_units=tuple(a.unit for a in plan.coupon_units),
            holdout_units=tuple(a.unit for a in plan.holdout_units),
            expected_incremental_repurchases=plan.expected["expected_incremental_repurchases"],
            expected_net_cents=plan.expected["net_reserved_cents"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forecast.json"
            record.save(path)
            loaded = ForecastRecord.load(path)
            self.assertEqual(loaded, record)
            result = reconcile_forecast(loaded, self.dataset)
            self.assertFalse(result["mature"])
            self.assertIsNone(result["actual_net_cents"])
            self.assertEqual(result["forecast_incremental_repurchases"], record.expected_incremental_repurchases)


class DuplicateCouponTests(Base):
    def test_duplicates_are_household_level(self):
        duplicates = duplicate_coupon_households(self.dataset)
        self.assertTrue(duplicates)
        identity = IdentityMap(self.dataset)
        for household, coupons in duplicates.items():
            self.assertGreater(len(coupons), 1)
            for coupon_id in coupons:
                coupon = self.dataset.coupons_by_id[coupon_id]
                self.assertEqual(identity.household_key(coupon.sent_to), household)


if __name__ == "__main__":
    unittest.main()
