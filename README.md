# 果箱电商留存投入

账户与订单使用无个人信息的合成代号。原始资料无电话、地址或真实券。

`data` 保存原始记录，`DATA.md` 是资料字典；`retention` 是分析包；`reports` 存放生成的结果。

环境使用 Python 3.11.2。安装锁定的依赖并运行检查：

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

## 生成结果

```sh
python3 -m retention --data data analyse --out reports/retention.md
python3 -m retention --data data bindings --out reports/late_bindings.json
python3 -m retention --data data plan --out reports/plan.json
python3 -m retention --data data reconcile --forecast reports/retention.forecast.json \
    --out reports/reconciliation.json
```

`analyse` 同时写出：

- `reports/retention.md`：四部分结论（首购队列、历史发券、时间外验证、本周建议、晚绑定）；
- `reports/retention.plan.json`：逐主体联系名单、排除原因、券预算与排程；
- `reports/retention.forecast.json`：**冻结**的预测。下一期用 `reconcile` 与实际兑换、成熟复购比对，预测文件本身不改写。

`plan` 提交本周客群、预算、券有效窗和每天接单能力；不传时读取 `data/policy.json` 的运营承诺。

## 分析口径（代码里的硬规则）

| 问题 | 规则 | 位置 |
|---|---|---|
| 谁算一个客户 | 只按 `confirmed_payer` 合并；空值不当作新主体；`仅同址` 不合并户 | `retention/identity.py` |
| 什么算一次购买 | `完成` 且非补发/指向原单；`全退`、`补发` 剔除 | `retention/orders.py` |
| 何时可用 | `returns_known_at` 已过；未成熟记“待定”，不记零 | `retention/orders.py` |
| 首购队列 | 每付款主体一笔；受赠人不算，除非本人另行付款 | `retention/orders.py` |
| 周间差异 | 构成 / 同类 / 交互 / 支撑外，四项精确加回 | `retention/cohorts.py` |
| 券效应 | 仅随机留量；报名单列；户内混合分配剔除；可转让券单列 | `retention/experiments.py` |
| 兑换 | 是结果不是处理，不做“已兑 vs 未兑”因果比较 | `retention/experiments.py` |
| 本周名单 | 确认付款人、有联系权限、无未兑券、同户去重、排除窗口内已回购 | `retention/campaign.py` |
| 留量 | 按渠道×首购场景×最近购买间隔分层，保留承诺份额 | `retention/campaign.py` |
| 晚绑定 | 已兑不撤；退出对照；后续联系只留一次；报告被挤占预算 | `retention/campaign.py` |
| 未登录访客 | 只报整体留存范围，永不进入联系名单 | `retention/campaign.py` |
| 预测修正 | 冻结预测文件 + 实际兑换/成熟回购对账，不用新分数改写旧活动 | `retention/campaign.py` |

检查只读取本地数据，不访问业务系统。
