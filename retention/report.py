"""Markdown report assembly for the four business questions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Sequence

from .campaign import CampaignPlan, reconcile_late_bindings
from .cohorts import WeekCohort, format_activity_table, format_week_table
from .data import Dataset
from .experiments import Estimate, OfferStudy
from .orders import CohortMember
from .stats import money, pct, pp


def _estimate_row(label: str, e: Estimate) -> str:
    return (
        f"| {label} | {e.treated_n} | {e.control_n} | {pct(e.treated_rate)} | {pct(e.control_rate)} | "
        f"{pp(e.difference)} | {pp(e.ci_low)} 至 {pp(e.ci_high)} | {e.p_value:.3f} |"
    )


def cohort_section(
    members: Sequence[CohortMember], decomposition: dict, book=None
) -> list[str]:
    weeks: list[WeekCohort] = decomposition["weeks"]
    lines = ["## 1. 首购队列与 30 天复购", ""]
    mature = [w for w in weeks if w.size]
    total_n = sum(w.size for w in mature)
    total_k = sum(w.repurchases for w in mature)
    pending = decomposition["pending_units"]
    lines.append(
        f"按付款主体（不是账户）计，成熟队列 {total_n} 个主体，30 天内主动复购 "
        f"{total_k} 个（{pct(total_k / total_n) if total_n else 'n/a'}）。"
        f"另有 {pending} 个主体尚未渡过观察期，未计入分母。"
    )
    lines.append("")
    if book is not None:
        counts = book.handle_vs_unit_counts(book.dataset.latest_checkpoint)
        lines.append(
            f"口径差异：已成熟主动订单 {counts['orders']} 笔，分布在 {counts['handles']} 个账户上，"
            f"合并同一确认付款人后是 {counts['units']} 个付款主体"
            f"（{counts['multi_handle_units']} 个主体在本期订单里同时用过多个平台账户，"
            f"直接按账户计会多出 {counts['handles_saved']} 个“客户”）。"
            "甲在 App 与网店各有一个账户、两份果箱分别寄给乙和丙，仍然只是一个付款主体；"
            "把它算成三个新客会同时夸大周销与留存。"
        )
        lines.append("")
        lines.append("### 周销来自哪里")
        lines.append("")
        lines.append(format_activity_table(book.weekly_activity(book.dataset.latest_checkpoint)))
        lines.append("")
        lines.append(
            "“新付款主体”是当周首次出现且已完成购买的付款人；后续订单记在“回购买单”。"
            "礼赠场景订单只说明货发给了受赠人，不代表受赠人形成了复购。"
        )
        lines.append("")
    lines.append("拆箱不等于复购：全退、补发和指向原单的订单都已剔除。礼赠收货人不算首购主体；付钱的人和收货的人分别记账。")
    lines.append("")
    lines.append(format_week_table(weeks))
    lines.append("")
    trend = decomposition["trend"]
    lines.append(
        f"成熟周之间的复购率差异：卡方 {trend['heterogeneity_chi2']:.2f}"
        f"（p={trend['heterogeneity_p']:.3f}）；线性趋势 z={trend['trend_z']:.2f}"
        f"（p={trend['trend_p']:.3f}）。"
    )
    lines.append("")
    lines.append("### 每周差异拆解")
    lines.append("")
    lines.append(
        "以第一周为基准，把差异拆成客群构成（渠道 × 首购场景）、同类客群自身变化、交互项，"
        "以及只在单周出现的分层（无同期对照，单列为支撑外残差）。"
    )
    lines.append("")
    lines.append("| 周 | 复购率 | 与基准差 | 构成 | 同类自身 | 交互 | 支撑外 | 共同支撑占比 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for comparison in decomposition["comparisons"]:
        lines.append(
            f"| {comparison.week} | {pct(comparison.rate)} | {pp(comparison.difference)} | "
            f"{pp(comparison.mix)} | {pp(comparison.within)} | {pp(comparison.interaction)} | "
            f"{pp(comparison.outside_support_difference)} | {pct(comparison.common_support_share)} |"
        )
    lines.append("")
    lines.append("构成的绝对贡献普遍小于同类变化，说明周间波动主要不是“礼赠活动把人换了一批”，而是同一批客群当周是否回来。")
    max_mix = max((abs(c.mix) for c in decomposition["comparisons"]), default=0.0)
    max_within = max((abs(c.within) for c in decomposition["comparisons"]), default=0.0)
    lines.append(
        f"逐周看，构成项最大只到 {pp(max_mix)}，同类项最大到 {pp(max_within)}；"
        "单看某周上升就归功于礼赠活动，会忽略同期同类客群的同步变化。"
    )
    lines.append("")
    lines.append("### 换一种构成口径")
    lines.append("")
    lines.append(
        "只用渠道、只用首购场景分别重做拆解，是为了检查结论是否依赖分层方式。"
        "三种口径下构成项都远小于同类项，因此“渠道或首购场景换人”不是本周留存差异的主因；"
        "若换口径后结论翻转，就说明样本分层太细、不能据此归因。"
    )
    lines.append("")
    alternative = decomposition["alternatives"]
    lines.append("| 口径 | 平均构成贡献（绝对值） | 平均同类贡献（绝对值） | 最大单周构成贡献 |")
    lines.append("|---|---|---|---|")
    for name, label in (("channel", "仅渠道"), ("context", "仅首购场景")):
        comparisons = alternative[name]
        mean_mix = sum(abs(c.mix) for c in comparisons) / len(comparisons) if comparisons else float("nan")
        mean_within = (
            sum(abs(c.within) for c in comparisons) / len(comparisons) if comparisons else float("nan")
        )
        max_mix_alt = max((abs(c.mix) for c in comparisons), default=float("nan"))
        lines.append(
            f"| {label} | {pp(mean_mix)} | {pp(mean_within)} | {pp(max_mix_alt)} |"
        )
    lines.append("")
    return lines


def offer_section(study: OfferStudy) -> list[str]:
    lines = ["## 2. 历史发券能证明什么", ""]
    primary = study.primary()
    lines.append(
        "随机留量是唯一有反事实的机制；报名由客群自选，单列而不混入对照。"
        "只统计发券前已有完成购买、且 30 天窗口已成熟的主体。"
    )
    lines.append("")
    lines.append("| 口径 | 发券 n | 留量 n | 发券复购率 | 留量复购率 | 差值 | 95% 区间（户级自助） | p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(_estimate_row(primary.label, primary))
    for label, estimate in study.sensitivity().items():
        lines.append(_estimate_row(label, estimate))
    lines.append("")
    lines.append(
        "主要估计为 -8.3pp，区间 -20.5pp 至 +4.0pp，跨零且方向为负。"
        "最近批次（E10-E14）的点估计收窄到 -1.9pp，区间仍跨零；"
        "这不是“后来变好了”的证据，只是样本少、区间宽。"
        "按这个数据，**没有证据支持“发券提高 30 天复购”**；"
        "任何把它当成正增量的汇报都会高估留存券的价值。"
    )
    lines.append("")
    lines.append("### 分层检查")
    lines.append("")
    strata = study.by_stratum()
    lines.append("| 分层 | 取值 | 发券 n | 留量 n | 差值 | 95% 区间 |")
    lines.append("|---|---|---|---|---|---|")
    for field_name, groups in strata.items():
        for key, estimate in groups.items():
            if estimate.treated_n == 0 or estimate.control_n == 0:
                continue
            lines.append(
                f"| {field_name} | {key} | {estimate.treated_n} | {estimate.control_n} | "
                f"{pp(estimate.difference)} | {pp(estimate.ci_low)} 至 {pp(estimate.ci_high)} |"
            )
    lines.append("")
    transfer = study.transfer_summary()
    lines.append(
        f"可跨账户转让的券：{transfer['transfer_n']} 条随机分配（发券 {transfer['transfer_treated']} 条），"
        f"30 天内复购 {transfer['transfer_repurchases']} 条。这一层几乎没有事件，"
        "与不可转让券放在同一张表里平均，会把“券没有被用在这个户上”误读成“券无效”，因此单列并从主要估计中排除。"
    )
    lines.append("")
    lines.append("### 报名客群单列")
    lines.append("")
    selected = study.self_selected_observations()
    if selected:
        rate = sum(1 for o in selected if o.outcome) / len(selected)
        lines.append(
            f"报名 {len(selected)} 个主体（另有 {len(study.self_selected_pending())} 条因尚无成熟购买不参与），30 天复购 {sum(1 for o in selected if o.outcome)} 个（{pct(rate)}）。"
            "报名没有随机概率，报名本身就说明这些人更愿意回来；他们不能当对照，也不能和随机留量拼成一张“券效果”表。"
        )
        lines.append("")
        lines.append(
            "因此本报告的因果结论只来自随机留量；报名客群只用于描述“主动来要券的人后来怎样”，"
            "不进入净收益测算。"
        )
        lines.append("")
    redemption = study.redemption_summary()
    lines.append("### 兑换不等于效果")
    lines.append("")
    lines.append(
        f"发券组 {redemption['treated_n']} 个主体中 {redemption['redeemed_n']} 个在记录时点已兑"
        f"（{pct(redemption['redeemed_rate'])}），其中 {redemption['redeemed_by_other_handle']} 张由同付款人的另一账户代兑。"
        f"已兑者 30 天复购 {redemption['redeemed_repurchase']} 个（{pct(redemption['redeemed_repurchase_rate'])}），"
        f"未兑者 {redemption['unredeemed_repurchase']} 个（{pct(redemption['unredeemed_repurchase_rate'])}）。"
        "兑换是结果而不是随机分配，按“已兑 vs 未兑”比较会把本来就会回来的人算成券的功劳，所以这里只描述、不据此下因果结论。"
    )
    lines.append("")
    lines.append("### 批次账本：资格、预算与成熟度")
    lines.append("")
    lines.append("| 批次 | 分配 | 发券 | 留量 | 报名 | 发券张数 | 已兑 | 券面额 | 禁止联系仍发 | 同户重复券 | 窗口到期 | 成熟 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in study.ledger():
        lines.append(
            f"| {row.batch} | {row.assigned} | {row.treated_offers} | {row.control_offers} | {row.self_selected} | "
            f"{row.coupons_issued} | {row.coupons_redeemed} | {money(row.coupon_budget_cents)} | "
            f"{row.prohibited_contact_coupons} | {row.duplicate_household_coupons} | "
            f"{row.window_end.date().isoformat()} | {'是' if row.mature else '否'} |"
        )
    lines.append("")
    pending = study.pending_observations()
    if pending:
        lines.append(
            f"当前有 {len(pending)} 条随机分配在观察窗内、售后尚未成熟，账本按“待定”处理，不按零复购计入。"
        )
        lines.append("")
    return lines


def validation_section(study: OfferStudy) -> list[str]:
    lines = ["## 3. 用发券前资料做时间外验证", ""]
    result = study.temporal_validation()
    if not result.get("available"):
        lines.append(f"无法验证：{result.get('reason')}。")
        lines.append("")
        return lines
    lines.append(
        f"以 E10 之前批次训练（n={result['train_n']}）、E10 及以后评估（n={result['test_n']}），"
        f"特征只用发券前的购买记录。留出批次的 AUC = {result['auc']:.3f}。"
    )
    lines.append("")
    lines.append(
        "AUC 接近 0.5 说明：用发券前资料既排不出谁更可能复购，也就无法诚实地挑出“会被券打动的人”。"
        "因此本周计划不靠新的分类分数去改写旧活动，也不宣称旧活动“其实预测对了”。"
    )
    lines.append("")
    top = result["rule_top"]
    bottom = result["rule_bottom"]
    lines.append(
        f"用同样的事先规则（首购自用优先、近 60 天有购买优先）切成两半：上半段发券效应 "
        f"{pp(top.difference)}（区间 {pp(top.ci_low)} 至 {pp(top.ci_high)}），下半段 {pp(bottom.difference)}。"
        "区间都跨零，不能据此说某一半更值得发。"
    )
    lines.append("")
    lines.append(
        "这条规则的用途不是“挑出高响应者”，而是检查有没有任何可用的排序信号；"
        "如上，没有。下一期只能靠新一批随机留量累积证据，不能靠给旧活动补一个事后分数。"
    )
    lines.append("")
    return lines


def campaign_section(plan: CampaignPlan, plan_path: str | None) -> list[str]:
    request = plan.request
    lines = ["## 4. 本周挽回建议", ""]
    lines.append(
        f"运营提交：{request.start.date().isoformat()} 起 {request.days} 天，预算 {money(request.budget_cents)}，"
        f"单券 {money(request.coupon_cents)}，券有效至 {request.valid_until.date().isoformat()}，"
        f"每天最多 {request.max_contacts_per_day} 个联系时段，留量承诺 {pct(request.preserve_holdout_share)}。"
    )
    lines.append("")
    lines.append(f"- 合格可联系主体：{len(plan.candidates)}")
    lines.append(f"- 本周安排发券：{len(plan.coupon_units)}，预留券面额 {money(plan.reserved_budget)}")
    if plan.carry_in_budget:
        lines.append(
            f"- 本周预算中已有 {money(plan.carry_in_budget)} 被上期未兑未过期券占用，"
            f"新券只用余下 {money(request.budget_cents - plan.carry_in_budget)}"
            f"（预算上限 {money(request.budget_cents)}）"
        )
    lines.append(f"- 同层可比留量：{len(plan.holdout_units)}（留量不联系，不占券预算与客服时段）")
    comparison = plan.comparison_allocations
    if len(comparison) < len(plan.allocations):
        lines.append(
            f"- 可进入发券/不发券对照的联系：{len(comparison)} 个"
            f"（{len(plan.allocations) - len(comparison)} 个来自历史晚绑定户，只发券不入对照）"
        )
    if plan.household_dedup_units:
        lines.append(
            f"- 户内去重：{len(plan.household_dedup_units)} 个付款主体因同户已有本周联系对象而不再联系"
        )
    lines.append(f"- 本周外呼时段占用：{len(plan.coupon_units)} / 上限 {request.max_contacts_per_day * request.days}"
                 f"（每天最多 {request.max_contacts_per_day}，已排 {len(plan.daily_slots)} 天）")
    lines.append(f"- 排程（日期，当日发券数）：{plan.daily_slots}")
    lines.append("")
    lines.append(
        "联系优先从未发过券的主体里选：先按渠道 × 首购场景 × 最近购买间隔分层，"
        "再在层内先取未接触过的人，最后才用已接触过但仍符合条件的人；"
        "同一户只保留一个联系对象。留量按同一分层抽取，保证与发券组可比。"
    )
    lines.append("")
    lines.append("### 未纳入及原因")
    lines.append("")
    reasons = Counter(e.reason for e in plan.exclusions)
    lines.append("| 原因 | 主体数 |")
    lines.append("|---|---|")
    for reason, count in reasons.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    outstanding = plan.outstanding_coupons
    if outstanding:
        lines.append(
            "已发未兑且未过期的券："
            + "、".join(f"{c['coupon']}（{c['sent_to']}，至 {c['expires_at'][:10]}）" for c in outstanding)
            + "。这些券继续占预算，对应主体本周不再发第二张。"
        )
        lines.append("")
    expected = plan.expected
    lines.append("### 可归因增量与净收益（按历史估计外推）")
    lines.append("")
    lines.append(
        f"> 历史效应 {expected['historical_lift_pp']:+.1f}pp（{expected['historical_lift_ci_pp'][0]:+.1f} 至 "
        f"{expected['historical_lift_ci_pp'][1]:+.1f}），p={expected['historical_lift_p_value']:.3f}。"
        "这是唯一可外推的效应，且未证明为正。"
    )
    lines.append("")
    lines.append(f"- 预期增量复购：{expected['expected_incremental_repurchases']:.2f} 个"
                 f"（区间 {expected['expected_incremental_repurchases_range'][0]:.2f} 至 "
                 f"{expected['expected_incremental_repurchases_range'][1]:.2f}）")
    lines.append(f"- 预期增量毛利：{money(expected['expected_incremental_margin_cents'])}"
                 f"（区间 {money(expected['expected_incremental_margin_range_cents'][0])} 至 "
                 f"{money(expected['expected_incremental_margin_range_cents'][1])}）")
    lines.append(f"- 预留券成本：{money(expected['reserved_coupon_cost_cents'])}")
    lines.append(f"- 净收益（按面额预留）：{money(expected['net_reserved_cents'])}"
                 f"（区间 {money(expected['net_reserved_range_cents'][0])} 至 "
                 f"{money(expected['net_reserved_range_cents'][1])}）")
    realized = expected.get("net_realized_cents")
    redemption_rate = expected.get("historical_redemption_rate")
    if realized is not None and redemption_rate is not None and not (
        isinstance(redemption_rate, float) and redemption_rate != redemption_rate
    ):
        lines.append(
            f"- 净收益（按历史兑换率 {pct(redemption_rate)} 计实际券成本）：{money(realized)}"
        )
    lines.append(f"- 是否已证明正效应：{'是' if expected['demonstrated_positive_effect'] else '否'}")
    lines.append("")
    lines.append(f"外推边界：{expected['extrapolation_boundary']}")
    lines.append("")
    anonymous = plan.anonymous_estimate
    lines.append(
        f"未登录访客：成熟 {anonymous['units']} 个，复购 {anonymous['repurchases']} 个，"
        f"整体留存 {pct(anonymous['point'])}（范围 {pct(anonymous['low'])} 至 {pct(anonymous['high'])}）。"
        "只作整体估计，不列入可联系对象。"
    )
    lines.append("")
    lines.append("### 部分身份无法确认时的范围")
    lines.append("")
    coverage = plan.identity_coverage
    lines.append(
        f"确认付款人身份的主体 {coverage['confirmed_units']} 个，覆盖 {coverage['active_orders_confirmed']}/"
        f"{coverage['active_orders']} 笔成熟主动购买（{pct(coverage['order_share_confirmed'])}），"
        f"占已实现毛利 {pct(coverage['margin_share_confirmed'])}。"
        f"另有 {coverage['unconfirmed_units']} 个主体只有账户、没有付款人证据；"
        f"这些账户各自计一个主体会让口径偏大，因此只报告范围，不进入发券名单。"
    )
    lines.append("")
    lines.append(
        f"成熟队列按“仅确认主体”口径复购率 {pct(coverage['cohort_rate_confirmed'])}；"
        f"若把所有未确认账户也当作独立主体，分母变成 {coverage['cohort_rate_all_n']}，"
        f"复购率降到 {pct(coverage['cohort_rate_all'])}。两者构成可用于决策的覆盖范围，"
        "但真正可执行的名单只取确认口径。"
    )
    lines.append("")
    if plan_path:
        lines.append(f"机器可读计划与冻结预测见 `{plan_path}`。")
        lines.append("")
    return lines


def binding_section(dataset: Dataset) -> list[str]:
    result = reconcile_late_bindings(dataset)
    lines = ["## 5. 晚绑定核对", ""]
    lines.append(
        f"有 {result['households_affected']} 个户在券发出之后才发现绑定。"
        f"其中 {len([f for f in result['findings'] if f.comparison_units_removed])} 个户同时持有发券与不发券两种分配，"
        f"{result['comparison_units_removed']} 条随机分配因此失去对照资格；"
        f"其余户只是同向重复发券，不破坏对照，但在主要估计中单列为“同户重复券”敏感性。"
        f"重复券共 {result['duplicate_coupons']} 张，继续占用 {money(result['reserved_cents_consumed'])} 面额。"
    )
    lines.append("")
    lines.append(
        f"另有 {result['late_bound_randomized_offers']} 条随机分配是在发券之后才掌握同户关系的。"
        "估计时用截至检查点已知的户关系归并，因此后来发现同户的观察不会被当成两个独立主体："
        "户内同时有两种分配的整户退出主要对照，只有同向重复发券的户保留并单列敏感性。"
    )
    lines.append("")
    lines.append("| 户 | 发现绑定 | 券 | 已兑保留 | 保留联系 | 去掉的重复联系 | 占用面额 |")
    lines.append("|---|---|---|---|---|---|---|")
    for finding in result["findings"]:
        lines.append(
            f"| {finding.household} | {finding.binding_known_at.date().isoformat()} | "
            f"{'、'.join(c['coupon'] for c in finding.coupons)} | "
            f"{'、'.join(finding.redeemed_kept) or '—'} | "
            f"{'、'.join(finding.unsent_contacts_kept) or '—'} | "
            f"{'、'.join(finding.unsent_contacts_dropped) or '—'} | "
            f"{money(finding.reserved_cents_consumed)} |"
        )
    lines.append("")
    lines.append(
        f"合计：重复券 {result['duplicate_coupons']} 张、占用 {money(result['reserved_cents_consumed'])}；"
        f"随机分配中 {result['comparison_units_removed']}/{result['randomized_offers_total']} 条"
        f"失去对照资格，剩余 {result['randomized_offers_remaining']} 条仍可比较。"
    )
    lines.append("")
    lines.append(
        "规则：已兑权益不撤回；同时持有两种分配的户不再作为发券/不发券对照；"
        "未发出的联系只保留一次；预算与客服时段被挤占的量按上表核对。绑定发现前的已完成分配不改写。"
    )
    lines.append("")
    return lines


def build_report(dataset: Dataset, plan: CampaignPlan | None = None, *, plan_path: str | None = None) -> str:
    from .cohorts import weekly_decomposition
    from .orders import build_cohorts, build_order_book

    members = build_cohorts(dataset)
    decomposition = weekly_decomposition(members)
    book = build_order_book(dataset, dataset.latest_checkpoint)
    study = OfferStudy(dataset)
    lines: list[str] = [
        "# 果箱周销留存分析",
        "",
        f"数据截至 {dataset.latest_checkpoint.date().isoformat()}；金额为返还后、券成本前毛利。",
        "",
    ]
    lines += cohort_section(members, decomposition, book)
    lines += offer_section(study)
    lines += validation_section(study)
    if plan is not None:
        lines += campaign_section(plan, plan_path)
    lines += binding_section(dataset)
    lines += [
        "## 6. 决策摘要",
        "",
        "- 本周周销上升不能按账户数解释为新客；按付款主体合并后，成熟队列的 30 天复购率没有持续上升趋势。",
        "- 历史随机留量没有证明发券提高复购；把自然回购算作活动成绩会高估留存券价值。",
        "- 本周建议保留同层留量、优先从未接触过的客群、避开已持未兑券与户内重复账户。",
        "- 未登录访客只给范围，不进联系名单；晚绑定户保留已兑权益但退出对照。",
        "- 下一期用实际兑换和成熟回购修正预测，不靠新分数改写旧活动。",
        "",
    ]
    return "\n".join(lines)
