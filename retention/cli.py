"""Command line entry point.

Usage::

    python3 -m retention analyse --data data --out reports/retention.md
    python3 -m retention plan --data data --out reports/plan.json
    python3 -m retention reconcile --data data --forecast reports/plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .campaign import (
    CampaignRequest,
    ForecastRecord,
    plan_campaign,
    reconcile_forecast,
    reconcile_late_bindings,
)
from .data import Dataset
from .report import build_report


def _request_from_policy(dataset: Dataset, *, start: datetime | None = None, days: int | None = None) -> CampaignRequest:
    policy = dataset.policy
    if start is None:
        # The plan can only use information frozen at the last checkpoint, so it
        # starts the following morning, never retroactively on the cut-off day.
        start = (dataset.latest_checkpoint + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    days = days or max(1, (policy.valid_until - start).days)
    return CampaignRequest(
        start=start,
        days=days,
        budget_cents=policy.budget_cents,
        coupon_cents=policy.coupon_cents,
        valid_until=policy.valid_until,
        max_contacts_per_day=policy.max_contacts_per_day,
        preserve_holdout_share=policy.preserve_holdout_share,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retention", description="果箱留存分析")
    parser.add_argument("--data", default="data", help="源数据目录")
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse", help="生成完整分析报告")
    analyse.add_argument("--data", default=argparse.SUPPRESS, help="源数据目录")
    analyse.add_argument("--out", default="reports/retention.md")
    analyse.add_argument("--start", help="本周计划开始时间（ISO，默认最后检查点）")
    analyse.add_argument("--days", type=int, help="计划天数（默认到券有效期）")

    plan = sub.add_parser("plan", help="生成本周联系计划与冻结预测")
    plan.add_argument("--out", default="reports/plan.json")
    plan.add_argument("--start", help="本周计划开始时间（ISO，默认最后检查点）")
    plan.add_argument("--days", type=int, help="计划天数")
    plan.add_argument("--budget-cents", type=int, help="覆盖预算")
    plan.add_argument("--coupon-cents", type=int, help="覆盖单券面额")
    plan.add_argument("--max-contacts-per-day", type=int, help="覆盖每日接单能力")

    reconcile = sub.add_parser("reconcile", help="用成熟结果核对冻结预测")
    reconcile.add_argument("--forecast", required=True, help="冻结预测 JSON")
    reconcile.add_argument("--out", default="reports/reconciliation.json")

    bindings = sub.add_parser("bindings", help="晚绑定核对")
    bindings.add_argument("--out", default="reports/late_bindings.json")

    args = parser.parse_args(argv)
    dataset = Dataset.load(args.data)

    if args.command == "analyse":
        start = datetime.fromisoformat(args.start) if args.start else None
        request = _request_from_policy(dataset, start=start, days=args.days)
        plan = plan_campaign(dataset, request)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plan_path = out.with_name(out.stem + ".plan.json")
        forecast_path = out.with_name(out.stem + ".forecast.json")
        plan_path.write_text(plan.to_json(), encoding="utf-8")
        record = ForecastRecord(
            campaign=f"week-{request.start.date().isoformat()}",
            made_at=request.start,
            eligibility_as_of=dataset.latest_checkpoint,
            budget_cents=request.budget_cents,
            coupon_cents=request.coupon_cents,
            valid_until=request.valid_until,
            contact_slots=len(plan.coupon_units),
            coupon_units=tuple(
                a.unit for a in plan.coupon_units if a.comparison_eligible
            ),
            holdout_units=tuple(
                a.unit for a in plan.holdout_units if a.comparison_eligible
            ),
            expected_incremental_repurchases=plan.expected["expected_incremental_repurchases"],
            expected_net_cents=plan.expected["net_reserved_cents"],
        )
        record.save(forecast_path)
        out.write_text(build_report(dataset, plan, plan_path=str(plan_path)), encoding="utf-8")
        print(f"报告: {out}")
        print(f"计划: {plan_path}")
        print(f"冻结预测: {forecast_path}")
        return 0

    if args.command == "plan":
        request = _request_from_policy(
            dataset,
            start=datetime.fromisoformat(args.start) if args.start else None,
            days=args.days,
        )
        if args.budget_cents or args.coupon_cents or args.max_contacts_per_day:
            request = CampaignRequest(
                **{
                    **request.__dict__,
                    "budget_cents": args.budget_cents or request.budget_cents,
                    "coupon_cents": args.coupon_cents or request.coupon_cents,
                    "max_contacts_per_day": args.max_contacts_per_day or request.max_contacts_per_day,
                }
            )
        plan = plan_campaign(dataset, request)
        record = ForecastRecord(
            campaign=f"week-{request.start.date().isoformat()}",
            made_at=request.start,
            eligibility_as_of=dataset.latest_checkpoint,
            budget_cents=request.budget_cents,
            coupon_cents=request.coupon_cents,
            valid_until=request.valid_until,
            contact_slots=len(plan.coupon_units),
            coupon_units=tuple(
                a.unit for a in plan.coupon_units if a.comparison_eligible
            ),
            holdout_units=tuple(
                a.unit for a in plan.holdout_units if a.comparison_eligible
            ),
            expected_incremental_repurchases=plan.expected["expected_incremental_repurchases"],
            expected_net_cents=plan.expected["net_reserved_cents"],
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        forecast_path = out.with_name(out.stem + ".forecast.json")
        out.write_text(plan.to_json(), encoding="utf-8")
        record.save(forecast_path)
        print(f"计划: {out}")
        print(f"冻结预测: {forecast_path}")
        return 0

    if args.command == "reconcile":
        record = ForecastRecord.load(args.forecast)
        result = reconcile_forecast(record, dataset)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "bindings":
        result = reconcile_late_bindings(dataset)
        payload = {
            key: value
            for key, value in result.items()
            if key != "findings"
        }
        payload["findings"] = [
            {
                "household": f.household,
                "binding_known_at": f.binding_known_at.isoformat(),
                "coupons": list(f.coupons),
                "redeemed_kept": list(f.redeemed_kept),
                "unsent_contacts_kept": list(f.unsent_contacts_kept),
                "unsent_contacts_dropped": list(f.unsent_contacts_dropped),
                "reserved_cents_consumed": f.reserved_cents_consumed,
                "contact_slots_consumed": f.contact_slots_consumed,
                "comparison_units_removed": list(f.comparison_units_removed),
                "note": f.note,
            }
            for f in result["findings"]
        ]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"晚绑定核对: {out}")
        return 0

    parser.error("unknown command")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
