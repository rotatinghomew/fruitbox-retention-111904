# 果箱电商留存投入

账户与订单使用无个人信息的合成代号。原始资料无电话、地址或真实券。

`data` 保存原始记录，`DATA.md` 是资料字典。这里只提供资料和格式检查；尚无分析、承诺或报表应用。

环境使用 Python 3.11.2。安装锁定的依赖并检查资料：

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

检查只读取本地数据，不访问业务系统。
