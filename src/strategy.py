import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from jqdata import *
from jqlib.technical_analysis import *

# 构建股票池
def get_security():
    # TODO
    # 返回一系列security的dataframe
    log.debug("executing get_security()")
    return

# 计算资金分配
def get_allocation():
    # TODO
    # 返回一系列对应股票顺序的比例（浮点数，总和为1）
    log.debug("executing get_allocation()")
    return

# 计算买入卖出信号
def get_signal():
    log.debug("executing get_signal()")
    buy_signal = False
    sell_signal = False
    # TODO
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
        g.security = get_security()
        g.allocation = get_allocation()
    g.period_counter += 1


## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))

    if g.security_changed:
        g.security_changed = False
        # sell all
        for each_security in g.security:
            order_target(each_security, 0)

    buy_signal, sell_signal = get_signal()

    if buy_signal:
        # TODO: buy
        pass
    elif sell_signal:
        # sell all
        for each_security in g.security:
            order_target(each_security, 0)


## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')
