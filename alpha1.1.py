import numpy as np
import math
import csv
from datetime import datetime
from scipy import stats
import matplotlib
import statsmodels.api as sm
import pickle
from matplotlib import pyplot as plt
mpl = matplotlib
# open data
data = np.genfromtxt('Class_game_test.csv', delimiter=',', unpack=True, filling_values="-9999",
                     dtype=None, missing_values="-9999", replace_space='_')
data2 = np.genfromtxt('RISK_FREE_CLASS_GAME.csv', delimiter=',', unpack=True, filling_values="-9999",
                      dtype=None, missing_values="-9999", replace_space='_')
tempdata = data
tempdata = np.transpose(tempdata)
colnames = tempdata[0]
exc_list = np.delete(colnames, 0, axis=0)
tempdata = np.delete(tempdata, 0, axis=0)
tempdata = np.transpose(tempdata)
data = tempdata

datespan = len(data[0])
dates = data[0]
temp2 = data2
temp2 = np.transpose(temp2)

temp2 = np.delete(temp2, 0, axis=0)
temp2 = np.transpose(temp2)
rf = temp2[1]


def cal_return_ini(date1, date0, etf):
    """ The function find price of etf at date 0 and date 1. Then compute simple rate of return
    with prices
    Args:
        date1(str): date t1
        date0(str): date t0
        etf(str):   the name of the ETF
    Returns:
        float: rate of return
    """
    # find data column
    for i in range(0, len(colnames)):
        if etf == colnames[i]:
            stock_val = data[i]
    # find price at t1
    for j in range(0, len(dates)):
        if date1 == (dates[j]):
            val1 = stock_val[j]
            val1 = float(val1)
    # find price at t0
    for m in range(0, len(dates)):
        if date0 == dates[m]:
            val0 = stock_val[m]
            val0 = float(val0)
    # find return
    ret = val1/val0 - 1
    return ret


def cal_return(date0, date1, etf, stocks_num):
    """The function returns simple return for a portfolio with number of different ETFs
    Arg:
        date0(str): date t0
        date1(str): date t1
        etf(list):  a list of ETFs in the portfolio
        stocks_num(list): a list of units of ETFs in the portfolio
    Returns:
        float: portfolio rate of return
    """
    ret = [0] * len(etf)
    val_port = [0] * len(etf)
    for k in range(0, len(etf)):
        for i in range(0, len(colnames)):
            if etf[k] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date0 == (dates[j]):
                        val1 = stock_val[j]
                        val1 = float(val1)
                        val_port[k] = val1
                for m in range(0, len(dates)):
                    if date1 == dates[m]:
                        val2 = stock_val[m]
                        val2 = float(val2)
                        ret[k] = val2/val1 - 1
    # find total value of portfolio
    val_tot = 0
    for i in range(0, len(stocks_num)):
        val_tot += val_port[i] * stocks_num[i]
    # find return of portfolio based on weight
    ret_p = 0
    for i in range(0, len(ret)):
        # return*weight
        ret[i] = (val_port[i] * stocks_num[i] * ret[i]) / val_tot
        ret_p += ret[i]
    return ret_p


def reb_amount(wealth, percent):
    """ The function calculate the re-balance amount as a percent of the
    total wealth of the portfolio
    Args:
        wealth(int):  wealth of the current portfolio
        percent(int): percentage
    Returns:
        int: amount to be re-balanced
    """
    money_buy = int(wealth * percent)
    return money_buy


def buy_stocks(etf, date1, cap):
    """The function buy a list of etf at date1 with capital in total.
    equally weighted
    Args:
        etf(list): a list of etf name
        date1(str): date to be bought
        cap(int): how much to buy in total
    Returns:
        (list): a list of units of each etf bought
        (float): capital left in hand
    """
    num_buy = [0] * len(etf)
    cap_left = [0] * len(etf)
    # find data, price
    for m in range(0, len(etf)):
        for i in range(0, len(colnames)):
            if etf[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == (dates[j]):
                        val1 = stock_val[j]
                        val1 = float(val1)
                        # number of units = (capital/number of ETF) / price
                        num_buy[m] = math.floor(cap / len(etf) / val1) - 1
                        # capital left with 0.1% transaction costs
                        cap_left[m] = cap / len(etf) - num_buy[m] * val1 * 1.001
    # find total amount of capital left
    cap_left_tot = 0
    for i in range(0, len(cap_left)):
        cap_left_tot += cap_left[i]
    return num_buy, cap_left_tot


def sell_stocks(etf, date1, num_to_sell, mon_left, money_buy, etf_buy):
    """The function sells the a list of existing ETF in the portfolio based on their weights
    in order to re-balance for other ETFs. If the ETF in the existing portfolio need to be
    bought later, then we do not sell it.
    Args:
        etf(list): a list of etf in the portfolio
        date1(str): trading date
        num_to_sell(list): a list of units of etf
        mon_left(float): value of money left
        money_buy(int): re-balance amount
        etf_buy(list): a list of etf to be bought later
    Return:
        (int): re-balance amount
        (list): units of each ETF left
        (float): capital exceeds re-balance amount after selling
    """
    cap_added = [0] * len(etf)
    num_to_sell_new = [0] * len(etf)
    val_stock = [0] * len(etf)
    val_stock_old = [0] * len(etf)
    stock_weights = [0] * len(etf)
    # if the ETF need to be bought
    # we do not sell it
    for i in range(0, len(etf)):
        if etf[i] == etf_buy:
            etf[i] = ''
    for i in range(0, len(etf)):
        for m in range(0, len(colnames)):
            if etf[i] == colnames[m]:
                stock_val = data[m]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        # past date, price from last period
                        val1 = stock_val[j-1]
                        val1 = float(val1)
                        val_stock_old[i] = val1
    tot_wealth = 0
    for i in range(0, len(etf)):
        tot_wealth += val_stock_old[i] * num_to_sell[i]
    # find weights of each ETFs
    for i in range(0, len(etf)):
        stock_weights[i] = (val_stock_old[i] * num_to_sell[i]) / tot_wealth
    for m in range(0, len(etf)):
        for i in range(0, len(colnames)):
            if etf[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        # put stock price in the list so we don't have to do the loops again
                        val_stock[m] = val1
                        # If number of stock sold > number of stock holding, we select number of stock we have
                        num_to_sell_new[m] = min((math.floor(money_buy * stock_weights[m]) / val1) + 1, num_to_sell[m])
                        cap_added[m] = num_to_sell_new[m] * val1 * 0.999
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot += cap_added[i]
    # find total money we have now
    ret = cap_added_tot + mon_left
    num_left = [0] * len(etf)
    # units of ETF left in the portfolio
    for i in range(0, len(num_to_sell)):
        num_left[i] = (num_to_sell[i] - num_to_sell_new[i])
    # while loop break until ret > money_buy
    # loop until we meet requirement - equally sell each ETF till the number of shares = 0
    while ret < money_buy:
        mon_need = money_buy - ret
        count = 0
        num_to_sell_again = [0] * len(etf)
        # find how many ETFs left in the portfolio
        for i in range(0, len(num_left)):
            if num_left[i] != 0:
                count += 1
        for j in range(0, len(num_left)):
            if num_left[j] != 0:
                # equally weighted sell the rest ETFs
                num_to_sell_again[j] = min(math.floor((mon_need / count) / val_stock[j]) + 1, num_left[j])
                cap_added[j] += num_to_sell_again[j] * val_stock[j] * 0.999
        for i in range(0, len(num_left)):
            num_left[i] -= num_to_sell_again[i]
        cap_added_tot = 0   # reset
        for i in range(0, len(cap_added)):
            cap_added_tot += cap_added[i]
        ret = cap_added_tot + mon_left
    money_left = ret - money_buy
    ret = money_buy
    return ret, num_left, money_left


def sell_stocks_all(etf, date1, num_to_sell, mon_left, etf_to_sell, money_buy):
    """The function sells everything units in the list of ETFs to sell
    Since it still could not cover the re-balance amount
    the functions then sells the rest ETFs in the portfolio
    based on their weights
    Args:
        etf(list): a list of etf in the portfolio
        date1(str): trading date
        num_to_sell(list): a list of units of etf
        mon_left(float): value of money left
        etf_to_sell(list): a list of etf to sell
        money_buy(int): re-balance amount
    Return:
        (int): re-balance amount
        (list): units of each ETF left
        (float): capital exceeds re-balance amount after selling
    """
    cap_added = [0] * len(etf_to_sell)
    val_stock = [0] * len(etf_to_sell)
    val_stock_last = [0] * len(etf_to_sell)
    for m in range(0, len(etf)):
        for i in range(0, len(colnames)):
            if etf_to_sell[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        val2 = stock_val[j-1]
                        val2 = float(val2)
                        # price of ETF this period
                        val_stock[m] = val1
                        # price of ETF last period
                        val_stock_last[m] = val2
                        # sell ETF in sell list
                        for n in range(0, len(etf)):
                            if etf_to_sell[m] == etf[n]:
                                cap_added[n] = val1 * num_to_sell[n] * 0.999
    # total amount of money get from sell ETF in sell list
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot += cap_added[i]
    ret = cap_added_tot + mon_left
    num_left = num_to_sell
    # replace units of ETF with 0 if ETF is in the sell ist
    for i in range(0, len(etf_to_sell)):
        for j in range(0, len(etf)):
            if etf[j] == etf_to_sell[i]:
                num_left[i] = 0
    # weights of left ETF according to price 1 day before the trading period
    t_wealth_test = 0
    for i in range(0, len(etf_to_sell)):
        t_wealth_test += num_left[i] * val_stock_last[i]
    weights_last = [0] * len(etf_to_sell)
    for i in range(0, len(weights_last)):
        weights_last[i] = (val_stock_last[i] * num_left[i]) / t_wealth_test
    # If total capital added < money we need
    # we sell more stocks if the number of stock left not equal to 0
    # while loop break until ret > money_buy
    # loop until we meet requirement - equally sell each ETF till the number of shares = 0
    while ret < money_buy:
        mon_need = money_buy - ret
        num_to_sell_again = [0] * len(etf)
        for j in range(0, len(num_left)):
            if num_left[j] != 0:
                # sell according to weights
                num_to_sell_again[j] = min(math.floor((mon_need * weights_last[j]) / val_stock[j]) + 1, num_left[j])
                cap_added[j] += num_to_sell_again[j] * val_stock[j] * 0.999
        for i in range(0, len(num_left)):
            num_left[i] -= num_to_sell_again[i]
        cap_added_tot = 0  # reset
        for i in range(0, len(cap_added)):
            cap_added_tot += cap_added[i]
        ret = cap_added_tot + mon_left
    money_left = ret - money_buy
    ret = money_buy
    return ret, num_left, money_left


def sell_stocks_sell(etf_to_sell, date1, num_to_sell, mon_left, money_buy):
    """The function sells ETFs in the sell list based on
    their weights; This can fulfill the requirement of re-balance amount
    Args:
        etf_to_sell(list): a list of etf to sell
        date1(str): trading date
        num_to_sell(list): a list of units of etf
        mon_left(float): value of money left
        money_buy(int): re-balance amount
    Return:
        (int): re-balance amount
        (list): units of each ETF left
        (float): capital exceeds re-balance amount after selling
    """
    cap_added = [0] * len(etf_to_sell)
    num_to_sell_new = [0] * len(etf_to_sell)
    val_stock = [0] * len(etf_to_sell)
    num_to_sell_again = [0] * len(etf_to_sell)
    val_stock_old = [0] * len(etf_to_sell)
    stock_weights = [0] * len(etf_to_sell)
    # find weights of each ETF in the sell list based on last period price
    for i in range(0, len(etf_to_sell)):
        for m in range(0, len(colnames)):
            if etf_to_sell[i] == colnames[m]:
                stock_val = data[m]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j-1]
                        val1 = float(val1)
                        val_stock_old[i] = val1
    tot_wealth = 0
    for i in range(0, len(etf_to_sell)):
        tot_wealth += val_stock_old[i] * num_to_sell[i]
    for i in range(0, len(etf_to_sell)):
        stock_weights[i] = (val_stock_old[i] * num_to_sell[i]) / tot_wealth
    # sell based on weights
    for m in range(0, len(etf_to_sell)):
        for i in range(0, len(colnames)):
            if etf_to_sell[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        # put stock price in the list so we don't have to do the loops again
                        val_stock[m] = val1
                        # If number of stock sold > number of stock holding, we select number of stock we have
                        num_to_sell_new[m] = min((math.floor((money_buy * stock_weights[m]) / val1) + 1),
                                                 num_to_sell[m])
                        cap_added[m] = num_to_sell_new[m] * val1 * 0.999
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot += cap_added[i]
    ret = cap_added_tot + mon_left
    num_left = [0] * len(etf_to_sell)
    num_left1 = [0] * len(etf_to_sell)
    # generate a list of units of ETF sold
    # use it at the end to find units of ETF left
    for i in range(0, len(etf_to_sell)):
        num_left1[i] = num_to_sell_new[i]
    # generate a list of units of ETF
    for i in range(0, len(num_to_sell)):
        num_left[i] = (num_to_sell[i] - num_to_sell_new[i])
    # replace units of ETF with 0 if this ETF is not in the sell list
    # units of ETF left of the sell list
    for j in range(0, len(num_to_sell)):
        for i in range(0, len(etf_to_sell)):
            if etf_to_sell[i] == '':
                num_left[i] = 0
    mon_need = money_buy - ret
    # If total capital added < money we need
    # we sell more stocks
    while ret < money_buy:
        if cap_added_tot < money_buy:
            stock_weights = [0] * len(etf_to_sell)
            tot_wealth = 0
            for i in range(0, len(etf_to_sell)):
                tot_wealth += val_stock_old[i] * num_left[i]
            # find weights
            for i in range(0, len(etf_to_sell)):
                stock_weights[i] = (val_stock_old[i] * num_left[i]) / tot_wealth
            # sell based on weights
            for j in range(0, len(num_left)):
                if num_left[j] != 0:
                    num_to_sell_again[j] += min(math.floor((mon_need * stock_weights[j]) / val_stock[j]) + 1,
                                                num_left[j])
                    cap_added[j] += num_to_sell_again[j] * val_stock[j] * 0.999
        cap_added_tot = 0
        for i in range(0, len(cap_added)):
            cap_added_tot += cap_added[i]
        ret = cap_added_tot + mon_left
    money_left = ret - money_buy
    ret = money_buy
    # find units of ETF sold for each ETFs in total
    for i in range(0, len(num_left)):
        num_left1[i] += num_to_sell_again[i]
    # find units of ETF left in the portfolio
    for i in range(0, len(num_left)):
        num_to_sell[i] -= num_left1[i]
    return ret, num_to_sell, money_left


def test_sell(etf_to_sell, date1, num_to_sell, money_left, money_buy):
    """The function sells all the ETFs in the sell list and check whether
    it can fulfill the re-balance amount requirement
    Args:
        etf_to_sell(list): a list of etf to sell
        date1(str): trading date
        num_to_sell(list): a list of units of etf
        money_left(float): value of money left
        money_buy(int): re-balance amount
    Return:
        (int): dummy = 1 [fulfill] and 2[cant fulfill]
    """
    cap_added = [0] * len(etf_to_sell)
    for m in range(0, len(etf_to_sell)):
        for i in range(0, len(colnames)):
            if etf_to_sell[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        for n in range(0, len(exc_list)):
                            if exc_list[n] == etf_to_sell[m]:
                                cap_added[m] = num_to_sell[n] * val1
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot += cap_added[i]
    tot = cap_added_tot + money_left
    # if can not fulfill, return 1
    if tot < money_buy:
        test = 1
    else:  # if can fulfill, return 2
        test = 2
    return test


def sell_all_sell_port(etf_to_sell, date1, num_to_sell, money_left, money_buy):
    """The function sells the ETFs in the list of sell
    However, there are no ETFs in the buy list
    So this function only sells all the ETFs in the sell list
    Args:
        etf_to_sell(list): a list of etf to sell in the portfolio
        date1(str): trading date
        num_to_sell(list): a list of units of etf
        money_left(float): value of money left
        money_buy(int): re-balance amount
    Return:
        (int):total money received
        (list):a list of units of etf
    """
    cap_added = [0] * len(etf_to_sell)
    num_to_sell_new = [0] * len(etf_to_sell)
    val_stock = [0] * len(etf_to_sell)
    val_stock_old = [0] * len(etf_to_sell)
    stock_weights = [0] * len(etf_to_sell)
    # find weights of each ETF in the sell list based on last period price
    for i in range(0, len(etf_to_sell)):
        for m in range(0, len(colnames)):
            if etf_to_sell[i] == colnames[m]:
                stock_val = data[m]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j - 1]
                        val1 = float(val1)
                        val_stock_old[i] = val1
    tot_wealth = 0
    for i in range(0, len(etf_to_sell)):
        tot_wealth += val_stock_old[i] * num_to_sell[i]
    for i in range(0, len(etf_to_sell)):
        stock_weights[i] = (val_stock_old[i] * num_to_sell[i]) / tot_wealth
    # sell based on weights
    for m in range(0, len(etf_to_sell)):
        for i in range(0, len(colnames)):
            if etf_to_sell[m] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        # put stock price in the list so we don't have to do the loops again
                        val_stock[m] = val1
                        # If number of stock sold > number of stock holding, we select number of stock we have
                        num_to_sell_new[m] = min((math.floor((money_buy * stock_weights[m]) / val1) + 1),
                                                 num_to_sell[m])
                        cap_added[m] = num_to_sell_new[m] * val1 * 0.999
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot += cap_added[i]
    ret = cap_added_tot + money_left
    num_left = [0]*len(etf_to_sell)
    # generate a list of units of ETF
    for i in range(0, len(num_to_sell)):
        num_left[i] = (num_to_sell[i] - num_to_sell_new[i])
    return ret, num_left


def sharpe_ratio(date1, weights, forward, interval):
    """The function calculates the sharpe ratio of the portfolio
    using the mean of past excess returns and std of past returns of the portfolio
    Args:
         date1(str): trading date
         weights(list): a list of weights of each etf
         forward(int):  regression interval(total)
         interval(int): interval between different dates
    Return:
    sharpe_ratio(float): Sharpe ratio
    var(float): variance
    """
    y_all = []
    y_all2 = []
    for m in range(0, len(exc_list)):
        yaxis = [0]*(forward / interval)
        yaxis2 = [0]*(forward / interval)
        etfs = exc_list[m]
        for i in range(0, len(dates)):
            if date1 == dates[i]:
                for j in range(0, len(yaxis)):
                    date_0 = dates[i - interval * (j+1)]
                    date_1 = dates[i - interval * j]
                    rf_t = float(rf[i - interval * (j+1)])
                    # return
                    yaxis[j] = cal_return_ini(date_1, date_0, etfs)
                    # excess return
                    yaxis2[j] = cal_return_ini(date_1, date_0, etfs) - rf_t

        y_all.append(yaxis)
        y_all2.append(yaxis2)
    exc_mean = []
    # expected excess return
    for i in range(0, len(y_all2)):
        r = np.mean(y_all2[i])
        exc_mean.append([r])
    # covariance matrix of return
    cov = np.cov(y_all)
    w_transpose = []
    # transpose the weights list
    for i in range(0, len(exc_list)):
        w_s = []
        w_s.append(weights[i])
        w_transpose.append(w_s)
    # find variance of portfolio = w*cov*w'
    var = np.dot(weights, cov)
    var = np.dot(var, w_transpose)
    # portfolio excess return
    excess_return = np.dot(weights, exc_mean)
    sharpe_r = excess_return / math.sqrt(var)
    return sharpe_r, var


def trading(date1, forward, interval, etf, stocks_in_hand, money_left, capital, percent, reb_style, fixed_amount):
    """The function taking asset allocations from last period and find alpha, making investment decisions and buy or sell
    securities
    Args:
         date1(str):          last trading date
         forward(int):        past interval for taking regression
         interval(int):       trading interval
         etf(list):           a list of etf in portfolio
         stocks_in_hand(list): a list of units of etfs
         money_left(float): cash in hand
         capital(float): wealth of portfolio
         percent(int):   percent of portfolio wealth if re-balance style is vary
         reb_style(str): re-balance style, vary or fixed
         fixed_amount(int):
    Return:
        etf_in_port(list): a list of etf in portfolio
        etf_num_port(list): a list of unit of etf in portfolio
        date_trade(str):   trading date
        total_mon_left(float): cash in hand
        t_wealth(float):    wealth of portfolio
        weight_port(list): a list of weights in portfolio
        buy(int):         buy decision dummy = 1 or 0
        sell(int):        sell decision dummy = 1 or 0
        sp(float):        Sharpe ratio
        var(float):       variance
        t_wealth_future(float): portfolio wealth in next trading date

    """
    v = 0
    for i in range(0, len(dates)):
        if dates[i] == date1:
            v = i
    date_trade = dates[v + interval]
    date_future = dates[v + 2 * interval]
    xaxis = [0]*(forward / interval)
    # define re-balance amount
    if reb_style == 'vary':
        money_buy = reb_amount(capital, percent)
    elif reb_style == 'fixed':
        money_buy = fixed_amount

    for i in range(0, len(dates)):
        if date_trade == dates[i]:
            for j in range(1, len(xaxis) + 1):
                date_0 = dates[i - interval * (j+1)]
                date_1 = dates[i - interval * j]
                rf_t = float(rf[i - interval * (j+1)])
                xaxis[j-1] = cal_return(date_0, date_1, etf, stocks_in_hand)-rf_t
    y_all = []
    for m in range(0, len(exc_list)):
        yaxis = [0]*(forward/interval)
        etfs = exc_list[m]
        for i in range(0, len(dates)):
            if date_trade == dates[i]:
                for j in range(1, len(yaxis)+1):
                    date_0 = dates[i - interval * (j+1)]
                    date_1 = dates[i - interval * j]
                    rf_t = float(rf[i - interval * (j+1)])
                    yaxis[j-1] = cal_return_ini(date_1, date_0, etfs)-rf_t
        y_all.append(yaxis)
    # find regression
    alpha_val = [0] * len(y_all)
    p_val = [0] * len(y_all)
    for i in range(0, len(y_all)):
        tempy = y_all[i]
        xaxis = sm.add_constant(xaxis)
        model = sm.OLS(tempy, xaxis)
        results = model.fit()
        alpha_val[i] = results.params[0]
        p_val[i] = results.pvalues[0]
    # find alpha>0 and p_value < 0.1
    etf_buy = []
    etf_sell = []
    for i in range(0, len(alpha_val)):
        if alpha_val[i] > 0 and p_val[i] < 0.1:
            etf_buy.append(exc_list[i])
        if alpha_val[i] < -0 and p_val[i] < 0.1:
            etf_sell.append(exc_list[i])

    if etf_buy != []:
        buy = 1
    else:
        buy = 0
    # check if ETF_sell is in the current portfolio
    sell_port = [''] * len(exc_list)
    if etf_sell != []:
        for i in range(0, len(etf_sell)):
            if etf_sell[i] not in etf:
                etf_sell[i] = ''
        for i in range(0, len(etf_sell)):
            for n in range(0, len(exc_list)):
                if exc_list[n] == etf_sell[i]:
                    sell_port[n] = etf_sell[i]
    if sell_port == ['']*len(exc_list):
        sell = 0
    else:
        sell = 1
    etf_s = [''] * len(exc_list)
    stocks_s = [0] * len(exc_list)
    for i in range(0, len(etf)):
        for j in range(0, len(exc_list)):
            if etf[i] == exc_list[j]:
                etf_s[j] = etf[i]
                stocks_s[j] = stocks_in_hand[i]
    # Sell ETFs to get fund for money_buy
    if sell_port == ['']*len(exc_list) and etf_buy != []:
        y4 = sell_stocks(etf_s, date_trade, stocks_s, money_left, money_buy, etf_buy)
        stocks_remain = y4[1]
    elif sell_port != ['']*len(exc_list) and etf_buy != []:
        # test to see the existing sell_list in my portfolio can fulfil the the requirements?
        # sell everything in (sell_list and also in my portfolio)
        y6 = test_sell(sell_port, date_trade, stocks_in_hand, money_left, money_buy)
        if y6 == 1:
            # cant, so we need to sell other ETFs that are not on the sell_list
            # sell them according to the weight
            y4 = sell_stocks_all(sell_port, date_trade, stocks_in_hand, money_left, etf, money_buy)
            stocks_remain = y4[1]
        if y6 == 2:
            # can
            # sell them according to the alpha weights
            y4 = sell_stocks_sell(sell_port, date_trade, stocks_in_hand, money_left, money_buy)
            stocks_remain = y4[1]
    elif sell_port != ['']*len(exc_list) and etf_buy == []:
        y6 = test_sell(sell_port, date_trade, stocks_in_hand, money_left, money_buy)
        if y6 == 1:
            # cannot
            # sell all the ETFs in Sell_port
            y4 = sell_all_sell_port(sell_port, date_trade, stocks_in_hand, money_left, money_buy)
            stocks_remain = y4[1]
        if y6 == 2:
            # can
            # sell ETFs according to the alpha weights
            y4 = sell_stocks_sell(sell_port, date_trade, stocks_in_hand, money_left, money_buy)
            stocks_remain = y4[1]
    elif etf_buy == [] and sell_port == [''] * len(exc_list):
        stocks_remain = stocks_in_hand
    # buy etf
    if etf_buy != []:
        y7 = buy_stocks(etf_buy, date_trade, money_buy)
    stocks_in_port = [0]*len(exc_list)
    # find units of ETFs in portfolio after buying and selling
    if etf_buy != []:
        for i in range(0, len(exc_list)):
            for j in range(0, len(etf_buy)):
                if exc_list[i] == etf_buy[j]:
                    stocks_in_port[i] = y7[0][j]
        for i in range(0, len(exc_list)):
            stocks_in_port[i] += stocks_remain[i]
    elif etf_buy == []:
        stocks_in_port = stocks_remain
    # find name of etf in portfolio
    etf_in_port = ['']*len(exc_list)
    for i in range(0, len(exc_list)):
        for j in range(0, len(exc_list)):
            if stocks_in_port[i] != 0:
                etf_in_port[i] = exc_list[i]
    etf_num_port = stocks_in_port
    # find total wealth and weight
    # find money left in cash
    if etf_buy != [] and sell_port == [''] * len(exc_list):
        total_mon_left = y4[2]+ y7[1]
    elif etf_buy != [] and sell_port != [''] * len(exc_list):
        total_mon_left = y4[2]+ y7[1]
    elif etf_buy == [] and sell_port != [''] * len(exc_list):
        if y6 == 1:
            total_mon_left = y4[0]
        elif y6 == 2:
            total_mon_left = money_left + money_buy
    elif etf_buy == [] and sell_port == [''] * len(exc_list):
        total_mon_left = money_left
    t_wealth = 0
    t_wealth_future = 0
    # find etf value
    val = [0]*len(exc_list)
    for k in range(0, len(etf_in_port)):
        for i in range(0, len(colnames)):
            if etf_in_port[k] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date_trade == dates[j]:
                        val1 = stock_val[j]
                        val1 = float(val1)
                        val[k] = val1
                        wealth = val1 * etf_num_port[k]
                        t_wealth += wealth
                    if date_future == dates[j]:
                        val2 = stock_val[j]
                        val2 = float(val2)
                        wealth2 = val2 * etf_num_port[k]
                        t_wealth_future += wealth2
    t_wealth += total_mon_left
    t_wealth_future = t_wealth_future + total_mon_left
    weight_port = [0]* len(etf_in_port)
    for i in range(0, len(etf_in_port)):
        weight_port[i] = round(float((val[i] * etf_num_port[i]) / t_wealth), 3)
    # find sharpe ratio
    sp, var = sharpe_ratio(date_trade, weight_port, forward, interval)
    return etf_in_port, etf_num_port, date_trade, total_mon_left, t_wealth, weight_port, buy, sell, sp, var, \
           t_wealth_future


def alpha(period, start_date, start_etf, forward, interval, percent, capital, reb_style, fixed_amount):
    """The function starts with the first trading date and buy stocks required
    then find next date wealth and enters in a loop to repeat the trading decisions
    Args:
        period(int):    how many loops
        start_date(str): starting date
        start_etf(list): a list of etf to buy at start
        forward(int):    past interval for taking regression
        interval(int):   trading interval
        percent(float):    re-balance amount as % of portfolio wealth
        capital(int):    capital to start with
        reb_style(str):  re-balance style:'vary' or 'fixed'
        fixed_amount(int): fixed amount of re-balance
    Return:
        wealth(float): wealth of the portfolio
    """
    wealth = []; weight1 = []; stock_num = []; time = []
    # tell me the date
    # program return whether its history horizon is suitable for regression
    for i in range(0, len(dates)):
        if start_date == dates[i]:
            if i < forward:
                print 'ERROR'
                break
    # buy stocks initially
    num_buy, cap_left_tot = buy_stocks(start_etf, start_date, capital)
    t_wealth_future = 0
    v = 0
    for i in range(0, len(dates)):
        if dates[i] == start_date:
            v = i
    date_future = dates[v + interval]
    # find next period wealth
    for k in range(0, len(num_buy)):
        for i in range(0, len(colnames)):
            if start_etf[k] == colnames[i]:
                stock_val = data[i]
                for j in range(0, len(dates)):
                    if date_future == dates[j]:
                        val2 = stock_val[j]
                        val2 = float(val2)
                        wealth2 = val2 * num_buy[k]
                        t_wealth_future += wealth2
    t_wealth_future = t_wealth_future + cap_left_tot
    # next period re-balance
    etf_in_port, etf_num_port, date_trade, total_mon_left, t_wealth, weight_port, buy, sell, sp, var, t_wealth_future\
    =trading(start_date, forward, interval, start_etf, num_buy, cap_left_tot, t_wealth_future, percent, reb_style,
             fixed_amount)
    wealth.append(math.floor(t_wealth))
    # loops for re-balance
    for p in range(0, period):
        etf_in_port, etf_num_port, date_trade, total_mon_left, t_wealth_future, weight_port, buy, sell, sp, var,\
        t_wealth_future = trading(date_trade, forward, interval, etf_in_port, etf_num_port, total_mon_left, t_wealth,
        percent, reb_style, fixed_amount)
        wealth.append(math.floor(t_wealth))
        print t_wealth_future
        print date_trade
    return wealth, stock_num, weight1, time, buy, sell, sp, var


if _name_ == "_main_":
    alpha = alpha(224, '31/01/96', ['SP500', 'Energy'], 60, 1,  0.05, 100000, 'vary', 7500)
