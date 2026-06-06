# ETF Rotation — A股 ETF 动量轮动系统

## 目标
基于 A 股 ETF 的动量信号，构建周频/月频轮动策略。

## 数据源
通过 Minishare API 获取数据：
- base_url: https://minishare.wmlgg.com
- API Key: 通过环境变量 `MINISHARE_API_KEY` 传入（macOS keychain: security find-generic-password -s "minishare" -a "MINISHARE_API_KEY" -w）
- ETF 日线: api_name=fund_daily, params={ts_code, trade_date, start_date, end_date}
- ETF 基础资料: api_name=etf_basic, params={market}
- A 股指数日线: api_name=index_daily

## 候选 ETF 池（主流宽基+行业）
- 510050.SH 上证50ETF
- 510300.SH 沪深300ETF
- 510500.SH 中证500ETF
- 159915.SZ 创业板ETF
- 588000.SH 科创50ETF
- 510880.SH 上证红利ETF
- 159949.SZ 创业板50ETF
- 159920.SZ 恒生ETF
- 159928.SZ 消费ETF
- 159929.SZ 医药ETF
- 512880.SH 证券ETF
- 159995.SZ 半导体ETF
- 515050.SH 5GETF
- 518880.SH 黄金ETF

## 工作内容
1. 搭建数据管道：从 Minishare 拉取 ETF 日线数据，存储本地
2. 动量计算：N 日收益率（20/60/120 日）+ 排名打分
3. 轮动策略：每月/每周按综合动量分选前 3-5 只
4. 历史回测：验证 2012-2025 年表现
5. 实盘信号：每周输出调仓建议
