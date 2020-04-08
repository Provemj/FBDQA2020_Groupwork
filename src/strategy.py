import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from jqdata import *
from jqlib.technical_analysis import *

'''
    构建股票池，策略参见文档
    返回的是一个股票代码的list，如['000632.XSHG', ...]
'''
def get_security(context):
    log.debug("executing get_security()")

    # 返回一系列security的dataframe

    security = []
    pre_dates = get_trade_days(start_date=None, end_date=context.current_dt, count=10)
    pre_date = pre_dates[0]
    pre_curr_date = pre_dates[-1]
    #INDUSTRY_CODES = ['J68']#, 'A01', 'HY524',]# 'HY491', 'GN736', 'GN815']
    INDUSTRY_CODES = ['801013', '801081', '801192', '801194', '801072', '801152']
    for industry_code in INDUSTRY_CODES:
        today = context.current_dt.date()
        stocks = get_industry_stocks(industry_code, date=today)


        #log.debug('stocks',stocks)
        # 十日跌幅前10%
        # TODO:这里的change_pct对吗
        change_pcts = []
        for stock in stocks:
            try:
                pre_data = get_price(stock, start_date=None, end_date=pre_date, frequency='daily', fields=['close'], skip_paused=True, fq='post', count=1,panel=False)
                pre_curr_data = get_price(stock, start_date=None, end_date=pre_curr_date, frequency='daily', fields=['close'], skip_paused=True, fq='post', count=1,panel=False)
                log.debug('pre_data',pre_data)
                log.debug('curr_data',pre_curr_data)
                pre_close = pre_data['close'].iloc[0]
                pre_curr_close = pre_curr_data['close'].iloc[0]
                change_pct = ( pre_close - pre_curr_close ) / pre_curr_close
                change_pcts.append(change_pct)
            except exceptions.ParamsError:
                log.error("找不到标的",stock)
                change_pcts.append(None)

        change_pct_df = pd.DataFrame({"code": stocks, "change_pct": change_pcts})
        change_pct_df.sort_values(by=['change_pct'], ascending=True, inplace=True)

        change_pct_df = change_pct_df.head(int(0.1 * len(change_pct_df))+1)
        log.debug('change_pct_df',change_pct_df.head())



        # TODO let it go或选取6只？
        # PB和市值
        pb_market_cap_df = get_fundamentals(query(
                valuation
            ).filter(
                # 在该板块内
                valuation.code.in_(stocks),
                # PB < 10
                valuation.pb_ratio < 10,
            ).order_by(
                valuation.market_cap.desc()
            ), date=today)
        # 市值在前90%
        pb_market_cap_df = pb_market_cap_df.head(int(0.9 * len(pb_market_cap_df)) + 1)
        log.debug('pb_market_cap_df',pb_market_cap_df.head())


        # 取交集 加入最终股票池
        for stock in stocks:
            if stock in pb_market_cap_df['code'].values and stock in change_pct_df['code'].values:
                security.append(stock)

    log.debug("return security\n", security)
    return security


# 计算资金分配
# 返回一系列对应股票顺序的比例（浮点数，总和为1）
def get_allocation(context):
    log.debug("executing get_allocation()")
    # TODO
    today = context.current_dt.date()
    stocks = g.security
    TREAD_DAY_COUNTS = 504
    # TODO: 参数设置？此处采用一整年的收盘价数据
    stock_prices_df = pd.DataFrame()
    for stock in stocks:
        price_df = get_price(stock, start_date=today + datetime.timedelta(days=-730), end_date=today, frequency='daily', fields=['close'])
        stock_prices_df[stock] = price_df['close'].values
    log.debug("stock_prices_df",stock_prices_df.head())

    # 计算收盘价数据的涨跌幅（收益率）
    stock_returns_df = stock_prices_df.pct_change().dropna()
    log.debug("stock_returns_df",stock_returns_df.head())

    # 相关系数
    correlation_matrix = stock_returns_df.corr()
    log.debug("correlation_matrix",correlation_matrix)

    # 协方差
    cov_mat = stock_returns_df.cov()
    log.debug("cov_mat",cov_mat)
    # 年化协方差矩阵
    cov_mat_annual = cov_mat * TREAD_DAY_COUNTS


    # 模拟次数
    SIMULATE_TIME = 10000

    # 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
    random_p = np.empty((SIMULATE_TIME, len(stocks) + 2))
    np.random.seed(0)

    # 循环模拟SIMULATE_TIME次随机的投资组合
    for i in range(SIMULATE_TIME):
        # 生成len(stocks)个随机数，并归一化，得到一组随机的权重数据
        random_len = np.random.random(len(stocks))
        random_weight = random_len / np.sum(random_len)

        # 计算年化平均收益率
        mean_return = stock_returns_df.mul(random_weight, axis=1).sum(axis=1).mean()
        annual_return = (1 + mean_return)**TREAD_DAY_COUNTS - 1

        # 计算年化的标准差，也称为波动率
        random_volatility = np.sqrt(np.dot(random_weight.T,
                                           np.dot(cov_mat_annual, random_weight)))

        # 将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
        random_p[i][:len(stocks)] = random_weight
        random_p[i][-2] = annual_return
        random_p[i][-1] = random_volatility

    # 将numpy数组转化成DataFrame数据框
    random_portfolios_df = pd.DataFrame(random_p)
    # 设置数据框RandomPortfolios每一列的名称
    random_portfolios_df.columns = [stock + "_weight" for stock in stocks] + ['Returns', 'Volatility']


    # TODO:设置无风险回报率为0 需要修改成10年债吗
    risk_free = 0
    random_portfolios_df['Sharpe'] = (random_portfolios_df.Returns - risk_free) / random_portfolios_df.Volatility

    # 找到夏普比率最大数据对应的索引值
    max_index = random_portfolios_df.Sharpe.idxmax()

    MSR_weights = np.array(random_portfolios_df.iloc[max_index, 0:len(stocks)])

    log.debug("return allocation", MSR_weights)
    log.debug("sum of allocation", MSR_weights.sum())
    return MSR_weights

# 计算买入卖出信号
def get_signal(context):
    log.debug("executing get_signal()")
    buy_signal = False
    sell_signal = False

	# 策略
    pool = ['801013.XSHG', '801081.XSHG', '801192.XSHG', '801194.XSHG', '801072.XSHG', '801152.XSHG']
    # 基准
    stock = '000300.XSHG'
    
    HS_da = get_price(security = stock, 
                      end_date = context.current_dt,
                      frequency = 'daily', 
                      fields = None, 
                      skip_paused = False, 
                      fq = 'pre',
                      count = 50)['close']
    # 双EXPMA 参考宽客示例
    EMA_da_2 = EMA(stock, check_date= HS_da.index[-2], timeperiod=30)[stock]
    EMA_da_1 = EMA(stock, check_date= HS_da.index[-1], timeperiod=5)[stock]
    
    # 短线上穿长线作买
    if HS_da[-2] < EMA_da_2 and HS_da[-1] > EMA_da_1:
        buy_signal=True
    # 短线下穿长线作卖
    elif HS_da[-2] > EMA_da_2 and HS_da[-1] < EMA_da_1: 
        sell_signal=True

    # 输出
    assert(buy_signal == False or sell_signal == False)
    return buy_signal, sell_signal

def initialize(context):
    ### 基础设置 ###
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    #log.set_level('order', 'error')


    ### 全局变量初始化 ###
    ## 默认每天运行，再平衡周期为10天
    g.period_counter = 0
    g.REBALANCE_PERIOD = 10
    g.security_changed = False


    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
    run_daily(market_open, time='open', reference_security='000300.XSHG')
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))

    ### 构建股票池 ###
    ## 十天为周期
    if g.period_counter % g.REBALANCE_PERIOD == 0:
        g.period_counter = 0
        g.security_changed = True
        # 再平衡
        g.security = get_security(context)
        g.allocation = get_allocation(context)
    g.period_counter += 1


## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))
    security=g.security
	# 取得当前的现金
    cash = context.portfolio.available_cash
	# 股票池更新
    if g.security_changed:
        g.security_changed = False
        # 空仓卖出
        for each in security:
            order_target(each, 0)

    buy_signal, sell_signal = get_signal(context)
    
    if buy_signal:
        # 按分配买入
        i=0
        for each in security:
			# 按比例矢量买入
            MSR_weights = get_allocation(context)
            order_value(each,MSR_weights[i]*cash)
            i=i+1
    elif sell_signal:
        # 空仓卖出
        for each in security:
            order_target(each, 0)


## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')
