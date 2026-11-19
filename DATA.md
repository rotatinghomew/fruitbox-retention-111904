# 资料字典

CSV 为 UTF-8，第一行为列名。日期为北京时间，时间带偏移；资料可见时刻与业务发生时刻分别以字段含义为准。空值不能默认做零。

### `accounts.csv`

主体身份有证据时使用 `confirmed_payer`，空值非确定新主体；操作权限按发券时有效记录。

字段：`handle`、`confirmed_payer`、`channel`、`contact_permission`、`recorded_at`。

### `bindings.csv`

`known_at` 表示掌握绑定的时间，`effective_from` 表示证据可追溯的开始日。共享地址不等于有效绑定；户级关系并不自动合并不同付款人。

字段：`left_handle`、`right_handle`、`household`、`known_at`、`evidence`、`effective_from`。

### `purchases.csv`

付款、礼赠收货、补发及原单分别表达；`returns_known_at` 以前售后尚未成熟。金额为返还后、券成本前毛利。

字段：`order`、`payer_handle`、`recipient_handle`、`source_context`、`paid_at`、`returns_known_at`、`outcome`、`replacement_of`、`margin_cents`。

### `offers.csv`

`随机留量` 分配单元是当时已确定的账户/户；户内代用可能破坏独立性。报名无随机概率。

字段：`offer`、`handle`、`batch`、`mechanism`、`assignment`、`treat_probability`、`assigned_at`、`transfer_possible`。

### `coupons.csv`

发券实际流水，留量不保证实际已发。空兑换人表示到记录时点未兑；已发券按面额保留预算。

字段：`coupon`、`offer`、`sent_to`、`cents`、`issued_at`、`expires_at`、`state`、`redeemed_by`。

### `policy.json`

留量份额指营运预先承诺的不发券比例。有限接单能力按联系时段占用，不主动联系匿名来客。
