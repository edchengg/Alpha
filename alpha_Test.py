import numpy as np
import math
import csv
from datetime import datetime
from scipy import stats
import matplotlib

mpl = matplotlib
from matplotlib import pyplot as plt

###open data
data = np.genfromtxt('Class_game_test.csv', delimiter=',', unpack=True, filling_values="-9999", dtype = None, missing_values = "-9999", replace_space='_')
data2 = np.genfromtxt('RISK_FREE_CLASS_GAME.csv', delimiter=',', unpack=True, filling_values="-9999", dtype = None, missing_values = "-9999", replace_space='_')
tempdata = data
tempdata = np.transpose(tempdata)
colnames = tempdata[0]
exc_list = np.delete(colnames, (0), axis = 0)
tempdata = np.delete(tempdata, (0), axis = 0)
tempdata = np.transpose(tempdata)
data = tempdata

datespan = len(data[0])
dates = data[0]
temp2 = data2
temp2 = np.transpose(temp2)

temp2 = np.delete(temp2,(0),axis = 0)
temp2 = np.transpose(temp2)
rf = temp2[1]
def cal_return_ini(date1, date0, ETF):

    for i in range(0, len(colnames)):
        if ETF == colnames[i]:
            stockval = data[i]
    for j in range(0, len(dates)):
        if date1 == (dates[j]):
            val1 = stockval[j]
            val1 = float(val1)
    for m in range(0, len(dates)):
        if date0 == dates[m]:
            val0 = stockval[m]
            val0 = float(val0)

    ret = val1/val0 - 1

    return ret
#test = cal_return_ini('14/03/13','14/03/12','STW')
#print test
#15.9253539697
def cal_return(date0, date1, ETF, stocksnum):

    ret = [0] * len(ETF)
    val_port = [0] * len(ETF)
    for k in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[k] == colnames[i]:
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date0 == (dates[j]):
                        val1 = stockval[j]
                        val1 = float(val1)
                        val_port[k] = val1
                for m in range(0, len(dates)):
                    if date1 == dates[m]:
                        val2 = stockval[m]
                        val2 = float(val2)
                        ret[k] = val2/val1 - 1
    ##find total value of portfolio
    val_tot = 0
    for i in range(0, len(stocksnum)):
        val_tot = val_tot + val_port[i] * stocksnum[i]
    ##find return of portfolio
    ret_p = 0
    for i in range(0, len(ret)):
        ret[i] = (val_port[i] * stocksnum[i] * ret[i]) / val_tot
        ret_p = ret_p + ret[i]

    return ret_p
#test = cal_return('14/03/12','14/03/13',['STW','SPDR50','SPDR200','','','','','','',''],[100,200,300,0,0,0,0,0,0,0])
#print test
#18.3834801841


capital = 100000
money_buy = 5000


def buy_stocks_ini(ETF, start_date, cap, forward):

    for i in range(0, len(colnames)):
        if ETF == colnames[i]:
            stockval = data[i]

    for j in range(0, len(dates)):
        if start_date == (dates[j]):
            k = j + forward
            val1 = stockval[k]
            val1 = float(val1)
    num_buy = math.floor(cap / val1)
    cap_left = (cap) - num_buy * (val1)

    return num_buy, cap_left
#test = buy_stocks_ini("SP500",'31/01/91',100000,50)
#print test
#(2116.0, 40.15999999998894)
def buy_stocks(ETF, date1, cap):
    num_buy = [0] * len(ETF)
    cap_left = [0] * len(ETF)

    for m in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[m] == colnames[i]:
                stockval = data[i]
                t_c = 0
                bas = 0
                for j in range(0, len(dates)):
                    if date1 == (dates[j]):
                        val1 = stockval[j]
                        val1 = float(val1)
                        num_buy[m] = math.floor((cap / (len(ETF))) / ((val1 + bas * val1) * (1 + t_c)))
                        cap_left[m] = (float(cap) / (len(ETF))) - num_buy[m] * val1
    cap_left_tot = 0
    for i in range(0, len(cap_left)):
        cap_left_tot = cap_left_tot + cap_left[i]

    return num_buy, cap_left_tot  ### Number of each stocks, money left
#test = buy_stocks(['STW','SPDR50','SPDR200'],'14/03/12',5000)
#print test
#([41.0, 40.0, 215.0], 13.809999999999945)

def sell_stocks_ini(ETF, date1, mon_left):
    cap_added = 0
    for i in range(0, len(colnames)):
        if ETF == colnames[i]:
            stockval = data[i]
            t_c = 0
            bas = 0
            for j in range(0, len(dates)):
                if date1 == (dates[j]):
                    val1 = stockval[j]
                    val1 = float(val1)
                    num_to_sell_new = math.floor((money_buy - mon_left) / ((val1 - bas * val1) * (1 - t_c)))
                    cap_added = (num_to_sell_new+1)*(1-bas)*val1*(1-t_c) #we sell one more stock
    return cap_added, num_to_sell_new+1 #inorder to meet $5000 requirement, we sell one more stock
#test = sell_stocks_ini('STW','14/03/12',10)
#print test
#(5020.76, 124.0)
def sell_stocks(ETF, date1, num_to_sell,mon_left,ETF_weights):
    #sell according to weights of ETFs
    cap_added = [0]*len(ETF)
    num_to_sell_new = [0]*len(ETF)
    val_stock = [0]*len(ETF)
    num_to_sell_again = [0] * len(ETF)


    for m in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[m] == colnames[i]:
                stockval = data[i]
                t_c = 0
                bas = 0
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        #put stock price in the list so we don't have to do the loops again
                        val_stock[m] = val1
                        #If number of stock sold > number of stock holding, we select number of stock we have
                        num_to_sell_new[m] = min((math.floor((money_buy*ETF_weights[m])/val1) + 1),num_to_sell[m])
                        cap_added[m] = num_to_sell_new[m]*(val1 - bas*val1)*(1-t_c)
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot = cap_added_tot + cap_added[i]
    ret = cap_added_tot + mon_left
    num_left = [0]*len(ETF)
    for i in range(0, len(num_to_sell)):
        num_left[i] = (num_to_sell[i] - num_to_sell_new[i])
    #while loop break until ret > money_buy
    #loop until we meet requirement - equally sell each ETF till the number of shares = 0
    while ret < money_buy:
        mon_need = money_buy - ret
        count = 0
        num_to_sell_again = [0] * len(ETF)
        for i in range(0, len(num_left)):
            if num_left[i] != 0:
                count = count + 1
        for j in range(0, len(num_left)):
            if num_left[j] != 0:
                num_to_sell_again[j] = min(math.floor((mon_need/count)/val_stock[j])+1,num_left[j])
                cap_added[j] = cap_added[j]+ num_to_sell_again[j]*val_stock[j]
        for i in range(0, len(num_left)):
            num_left[i] = num_left[i] - num_to_sell_again[i]
        cap_added_tot = 0 #reset
        for i in range(0, len(cap_added)):
            cap_added_tot = cap_added_tot + cap_added[i]
        ret = cap_added_tot + mon_left
    money_left = ret - money_buy
    ret = money_buy

    return ret, num_left, money_left
#test = sell_stocks(['SP500','Energy','','','','','','','','','','','','',''],'28/02/91',[9950,500,0,0,0,0,0,0,0,0,0,0,0,0],10,[0.95433,0.045667,0,0,0,0,0,0,0,0,0,0,0,0])
#print test
#(5000, [9523.0, 478.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 17.172452840000005)
def sell_stocks_all(ETF, date1, num_to_sell,mon_left,ETF_e):
    cap_added = [0]*len(ETF_e)
    num_to_sell_new = [0]*len(ETF_e)
    val_stock = [0]*len(ETF_e)
    num_to_sell_again = [0] * len(ETF_e)
    for m in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[m] == colnames[i]:
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        for n in range(0, len(ETF_e)):
                            if ETF[m] == ETF_e[n]:
                                val_stock[n] = val1
                                cap_added[n] = val1*num_to_sell[n]
    for m in range(0, len(ETF_e)):
        for i in range(0, len(colnames)):
            if ETF_e[m] == colnames[i]:
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val2 = stockval[j]
                        val2 = float(val2)
                        val_stock[m] = val2
    #print cap_added
    #print val_stock
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot = cap_added_tot + cap_added[i]
    #print cap_added_tot
    ret = cap_added_tot + mon_left
    num_left = num_to_sell
    for i in range(0, len(ETF_e)):
        for j in range(0, len(ETF)):
            if ETF[j] == ETF_e[i]:
                num_left[i] = 0
    #If total capital added < money we need
    #we sell more stocks if the number of stock left not equal to 0

    #while loop break until ret > money_buy
    #loop until we meet requirement - equally sell each ETF till the number of shares = 0
    while ret < money_buy:
        mon_need = money_buy - ret
        count = 0
        num_to_sell_again = [0] * len(ETF)
        for i in range(0, len(num_left)):
            if num_left[i] != 0:
                count = count + 1
        for j in range(0, len(num_left)):
            if num_left[j] != 0:
                num_to_sell_again[j] = min(math.floor((mon_need/count)/val_stock[j])+1,num_left[j])
                cap_added[j] = cap_added[j]+ num_to_sell_again[j]*val_stock[j]
        for i in range(0, len(num_left)):
            num_left[i] = num_left[i] - num_to_sell_again[i]
        cap_added_tot = 0 #reset
        for i in range(0, len(cap_added)):
            cap_added_tot = cap_added_tot + cap_added[i]
        ret = cap_added_tot + mon_left
    moneyleft = ret - money_buy
    ret = money_buy
    return ret, num_left, moneyleft

#test = sell_stocks_all(['','SPDR50','','','','','','','','RUSSELCPRTBD'],'14/03/12',[171,50,50,0,0,0,0,0,0,1],100,['STW','SPDR50','SPDR200','','','','','','','RUSSELCPRTBD'])
#print test
#(5000, [111.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 13.920000000000073)

def sell_stocks_sell(ETF, date1, num_to_sell,mon_left):
    cap_added = [0]*len(ETF)
    num_to_sell_new = [0]*len(ETF)
    val_stock = [0]*len(ETF)
    num_to_sell_again = [0] * len(ETF)
    count1 = 0
    for i in range(0, len(ETF)):
        if ETF[i] != '':
            count1 += 1
    for m in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[m] == colnames[i]:
                stockval = data[i]
                t_c = 0
                bas = 0
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        #put stock price in the list so we don't have to do the loops again
                        val_stock[m] = val1
                        #If number of stock sold > number of stock holding, we select number of stock we have
                        num_to_sell_new[m] = min((math.floor((money_buy/count1)/val1) + 1),num_to_sell[m])
                        cap_added[m] = num_to_sell_new[m]*(val1 - bas*val1)*(1-t_c)
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot = cap_added_tot + cap_added[i]
    ret = cap_added_tot + mon_left
    num_left = [0]*len(ETF)
    num_left1 = [0]*len(ETF)
    for i in range(0, len(ETF)):
        num_left1[i] = num_to_sell_new[i]
    #print num_left1
    for i in range(0, len(num_to_sell)):
        num_left[i] = (num_to_sell[i] - num_to_sell_new[i])
    for j in range(0, len(num_to_sell)):
        for i in range(0, len(ETF)):
            if ETF[i] == '':
                num_left[i] = 0
    #print num_left
    mon_need = money_buy - ret
    #If total capital added < money we need
    #we sell more stocks if the number of stock left not equal to 0
    if cap_added_tot < money_buy:
        count = 0
        for i in range(0, len(num_left)):
            if num_left[i] != 0:
                count = count + 1
        #print count
        #equally sell stocks if number of stocks not equal to 0
        for j in range(0, len(num_left)):
            if num_left[j] != 0:
                num_to_sell_again[j] = min(math.floor((mon_need/count)/val_stock[j])+1,num_left[j])
                cap_added[j] = cap_added[j]+ num_to_sell_again[j]*val_stock[j]

    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot = cap_added_tot + cap_added[i]
    ret = cap_added_tot + mon_left
    moneyleft = ret - money_buy
    ret = money_buy
    #find number of stocks left
    for i in range(0, len(num_left)):
        num_left1[i] = num_left1[i] + num_to_sell_again[i]
    #print num_left1
    for i in range(0, len(num_left)):
        num_to_sell[i] = num_to_sell[i] - num_left1[i]
    return ret, num_to_sell, moneyleft
#test = sell_stocks_sell(['','SPDR50','','','','','','','','RUSSELCPRTBD'],'14/03/12',[171,80,50,0,0,0,0,0,0,100],100)
#print test
#(5000, [171, 10.0, 50, 0, 0, 0, 0, 0, 0, 0], 10.5)
def test_sell(ETF,date1,num_to_sell,money_left):
    cap_added = [0]*len(ETF)
    val_stock = [0]*len(ETF)
    for m in range(0, len(ETF)):
        for i in range(0, len(colnames)):
            if ETF[m] == colnames[i]:
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date1 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        for n in range(0, len(exc_list)):
                            if exc_list[n] == ETF[m]:
                                cap_added[m] = num_to_sell[n]*val1
    cap_added_tot = 0
    for i in range(0, len(cap_added)):
        cap_added_tot = cap_added_tot + cap_added[i]
    #print cap_added_tot
    tot = cap_added_tot + money_left
    if tot < money_buy:
        test = 1
    else:
        test = 2
    return test

#test = test_sell(['STW','','SPDR200','','','','','','','RUSSELCPRTBD'],'14/03/12',[17,0,1000,0,0,0,0,0,0,50],100)
#print test
#2

def cal_reg_ini(start_date, forward, interval, start_ETF):
    y2 = buy_stocks_ini(start_ETF, start_date, capital, forward)
    #print y2[0]
    #print y2[1]
    #tell me data starting date, i give you the earliest trading date:start+forward
    stocks_ini = y2[0]
    mon_left_beg = y2[1]

    #next trading date is date1 + forward + interval
    for j in range(0, len(dates)):
        if start_date == dates[j]:
            date2 = dates[j+ forward + interval+1]
    #print date2
    #find x-axis and y-axis
    xaxis = [0]*((forward/interval)+1)
    for j in range(0, len(dates)):
        if start_date == (dates[j]):
            for i in range(0, len(xaxis)):
                date_mid = dates[j + interval * (i+1)]
                ini_date = dates[j + interval * (i)]
                rf_t = rf[j + interval *(i) ]
                rt = cal_return_ini(date_mid,ini_date,start_ETF)
                xaxis[i] = rt - float(rf_t)
    #print xaxis
    y_all = []
    for m in range(0, len(exc_list)):
        yaxis = []
        ETF = exc_list[m]
        for j in range(0, len(dates)):
            if start_date == dates[j]:
                for i in range(0, len(xaxis)):
                    date_mid = dates[j + interval * (i+1)]
                    last_date = dates[j + interval * (i)]
                    rf_t = float(rf[j + interval *(i) ])
                    yaxis.append(cal_return_ini(date_mid,last_date, ETF)-rf_t)
        y_all.append(yaxis)

    #extract rf rate

    #print y_all[0]
    #find regression
    alpha_val = [0] * len(y_all)
    p_val = [0] * len(y_all)
    for i in range(0, len(y_all)):
        tempy = y_all[i]
        slope, alpha, r_value, p_value, std_err = stats.linregress(xaxis, tempy)
        alpha_val[i] = alpha
        p_val[i] = p_value

    #find alpha>0 and p_value < 0.1

    ETF_buy = []

    for i in range(0, len(alpha_val)):
        if alpha_val[i] > 0.01 and p_val[i] < 0.1:
            ETF_buy.append(exc_list[i])
    #print ETF_buy
    #next trading day is date2
    #only have starting_ETF, so sell it
    if ETF_buy != '':

      y4 = sell_stocks_ini(start_ETF,date2,mon_left_beg)
      total_mon = y4[0] + mon_left_beg
      buy_cap = money_buy

      y = buy_stocks(ETF_buy,date2,buy_cap)
      num_bought = y[0]
      cap_rem = y[1]

      total_mon_left = cap_rem + total_mon - buy_cap

    ETF_in_port = ['']*len(exc_list)
    ETF_num_port = [0]*len(exc_list)
    num_of_stock = stocks_ini - y4[1]#number to sell
    if num_of_stock != 0:
        ETF_in_port[0] = exc_list[0]
        ETF_num_port[0] = num_of_stock
    #construct namelist of ETF in my portfolio, e.g.['VAS', 'VTS', 'STW', '', 'VEU', '']
    #construct number_list of ETF in my portfolio, e.g.[1549.0, 26.0, 39.0, 0, 35.0, 0]
    for i in range(0, len(exc_list)):
        for j in range(0, len(ETF_buy)):
            if exc_list[i] == ETF_buy[j]:
                ETF_in_port[i]= exc_list[i]
                ETF_num_port[i] = num_bought[j]

    #print ETF_in_port
    #print ETF_num_port
    #find total wealth and weight
    twealth = 0
    val = [0]*len(exc_list)
    for k in range(0, len(ETF_in_port)):
        for i in range(0, len(colnames)):
            if ETF_in_port[k] == colnames[i]:
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date2 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        val[k] = val1
                        wealth = val1*ETF_num_port[k]
                        twealth = wealth + twealth
    twealth = twealth + total_mon_left
    weight_port = [0]* len(ETF_in_port)
    for i in range(0, len(ETF_in_port)):
        weight_port[i] = round(float((val[i]*ETF_num_port[i])/twealth),3)
    return ETF_in_port, ETF_num_port, date2, total_mon_left,twealth,weight_port

#test = cal_reg_ini('14/03/12', 30,10,'STW')
#print test
#(['STW', '', '', '', 'RUSSELHIGHD', '', 'VANGDSHS', '', '', ''], [2304.0, 0, 0, 0, 108.0, 0, 50.0, 0, 0, 0], '10/5/2012', 39.75999999999112, 98737.95999999999, [0.949, 0.0, 0.0, 0.0, 0.025, 0.0, 0.025, 0.0, 0.0, 0.0])
def cal_reg(date1,forward,interval,ETF,stocks_inhand,money_left,weights_now):
    v = 0
    for i in range(0, len(dates)):
        if dates[i] == date1:
            v = i
    date2 = dates[v + interval]
    date3 = dates[v + interval + 1]
    xaxis = [0]*((forward/interval)+1)

    for i in range(0, len(dates)):
        if date2 == dates[i]:
            for j in range(0, len(xaxis)):
                date_0 = dates[i - interval*(j+1)]
                date_1 = dates[i - interval*j]
                rf_t = float(rf[i - interval * (j+1)])
                xaxis[j] = cal_return(date_0, date_1, ETF, stocks_inhand)-rf_t
    y_all = []
    for m in range(0, len(exc_list)):
        yaxis = [0]*((forward/interval)+1)
        ETFs = exc_list[m]
        for i in range(0, len(dates)):
            if date2 == dates[i]:
                for j in range(0, len(yaxis)):
                    date_0 = dates[i - interval * (j+1)]
                    date_1 = dates[i - interval * j]
                    rf_t = float(rf[i - interval *(j+1)])
                    yaxis[j] = cal_return_ini(date_1, date_0, ETFs)-rf_t
        y_all.append(yaxis)

    #find regression
    alpha_val = [0] * len(y_all)
    p_val = [0] * len(y_all)
    for i in range(0, len(y_all)):
        tempy = y_all[i]
        slope, alpha, r_value, p_value, std_err = stats.linregress(xaxis, tempy)
        alpha_val[i] = alpha
        p_val[i] = p_value######check
    #print alpha_val
    #print p_val
    #find alpha>0 and p_value < 0.1
    ETF_buy = []

    ETF_sell = []

    for i in range(0, len(alpha_val)):
        if alpha_val[i] > 0.01 and p_val[i] < 0.1:
            ETF_buy.append(exc_list[i])
        if alpha_val[i] < -0.01 and p_val[i] < 0.1:
            ETF_sell.append(exc_list[i])
    #print ETF_buy
    #print ETF_sell
    #ETF_sell = ['VTS','STW','SLF']
    #Check if ETF_sell is in the current portfolio
    Sell_port = ['']*len(exc_list)
    for i in range(0, len(ETF_sell)):
        for n in range(0, len(exc_list)):
            if exc_list[n] == ETF_sell[i]:
                Sell_port[n] = ETF_sell[i]
    for i in range(0, len(ETF)):
        for j in range(0, len(Sell_port)):
            if ETF[i] == Sell_port[j]:
                Sell_port[i] = ETF[i]

    #find ETF weights in current portfolio
    if Sell_port == ['']*len(exc_list) and ETF_buy != []:
        y4 = sell_stocks(ETF,date3,stocks_inhand,money_left,weights_now)
        print y4
    elif Sell_port != ['']*len(exc_list) and ETF_buy != []:

        y6 = test_sell(Sell_port,date3,stocks_inhand,money_left)
        if y6 == 1:
            y4 = sell_stocks_all(Sell_port,date3,stocks_inhand,money_left,ETF)
            print y4

        if y6 ==2:
            y4 = sell_stocks_sell(Sell_port,date3,stocks_inhand,money_left)
            print y4
    if ETF_buy == []:
        stocks_remain = stocks_inhand
    else:
        stocks_remain = y4[1]

    y7 = buy_stocks(ETF_buy,date3,money_buy)
    #print y7[0]
    stocks_in_port = [0]*len(exc_list)
    for i in range(0, len(exc_list)):
        for j in range(0, len(ETF_buy)):
            if exc_list[i] == ETF_buy[j]:
                stocks_in_port[i] = y7[0][j]
    #print stocks_in_port
    for i in range(0, len(exc_list)):
        stocks_in_port[i] = stocks_in_port[i]+ stocks_remain[i]
    #print stocks_in_port
    ETF_in_port = ['']*len(exc_list)
    ETF_num_port = [0]*len(exc_list)
    for i in range(0, len(exc_list)):
        for j in range(0, len(exc_list)):
            if stocks_in_port[i] != 0:
                ETF_in_port[i]= exc_list[i]
    #print ETF_in_port
    ETF_num_port = stocks_in_port
    #print ETF_num_port
    #find total wealth and weight
    if ETF_buy != []:
        total_mon_left = y4[2]+y7[1]
    else:
        total_mon_left = money_left
    twealth = 0
    val = [0]*len(exc_list)
    for k in range(0, len(ETF_in_port)):
        for i in range(0, len(colnames)):
            if ETF_in_port[k] == colnames[i]:
                #print ETF_in_port[k]
                #print colnames[i]
                stockval = data[i]
                for j in range(0, len(dates)):
                    if date3 == dates[j]:
                        val1 = stockval[j]
                        val1 = float(val1)
                        val[k] = val1
                        wealth = val1*ETF_num_port[k]
                        #print wealth
                        twealth = twealth + wealth
                        #print twealth
    twealth = twealth + total_mon_left#rf rate
    weight_port = [0]* len(ETF_in_port)

    for i in range(0, len(ETF_in_port)):
        weight_port[i] = round(float((val[i]*ETF_num_port[i])/twealth),3)

    return ETF_in_port, ETF_num_port, date3, total_mon_left,twealth,weight_port
#y= cal_reg('28/02/91',60,1,['SP500','Energy','','','','','','','','','','','','',''], [9500,500,0,0,0,0,0,0,0,0,0,0,0,0,0],30,[0.95202,0.047714,0,0,0,0,0,0,0,0,0,0,0,0,0])
#print y
#(['SP500', 'Energy', '', '', '', '', '', '', '', '', '', '', '', '', ''],
# [9084.0, 477.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '30/04/91',
# 47.775877469999614, 109348.115175, [0.953, 0.047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#check 120070 = weights* return price#self financing constraints
def Alpha(period,start_date,start_ETF,forward,interval):
    wealth = []
    weight1 = []
    stock_num = []
    time = []
    y0 = cal_reg_ini(start_date,forward,interval,start_ETF)
    wealth.append(math.floor(y0[4]))
    weight1.append((y0[5]))
    stock_num.append(y0[1])
    time.append(y0[2])
    print y0[4]
    print y0[2]
    y = cal_reg(y0[2],forward,interval,y0[0],y0[1],y0[3],y0[5])
    wealth.append(math.floor(y[4]))
    weight1.append((y[5]))
    stock_num.append((y[1]))
    time.append(y[2])
    print y[4]
    print y[2]
    for a in range(0, period):
       y = cal_reg(y[2],forward, interval, y[0],y[1],y[3],y[5])
       #print 'wealth:'
       print y[4]
       #print 'Date'
       print y[2]
       #print 'weight'
       print y[5]
       wealth.append(math.floor(y[4]))
       weight1.append((y[5]))
       stock_num.append((y[1]))
       time.append(y[2])

    return wealth,stock_num,weight1,time

#AAA = Alpha(50,'11/04/2007','SPDR500',500,10)
AAA = Alpha(110,'31/01/91','SP500',60,1)
wealth = AAA[0]
weights = AAA[2]
time = AAA[3]
stock_num = AAA[1]
print len(weights)
#compre it with SP500
v = 0
for i in range(0,len(dates)):
    if dates[i] == '29/03/96':
        stockval = data[1][i]
        stockval = float(stockval)
        v = i
stocknum = math.floor(100000/stockval)
sp_wealth = [0]*112
for i in range(0,112):
    sp_wealth[i] = float(data[1][v+i])*stocknum

def plot(time, stock_num, wealth, weights,sp_wealth):
 s1 =[]


 for i in range(0,len(stock_num)):
     s1.append(stock_num[i][0])


 x = range(len(s1))
 plt.plot(x,wealth)
 plt.plot(x,sp_wealth)
 plt.legend(['portfolio','SP500'],loc='upper right',fontsize = 'x-small')
 plt.show()



 w1 =[]
 w2 =[]
 w3 =[]
 w4 =[]
 w5 =[]
 w6 =[]
 w7 =[]
 w8 =[]
 w9 =[]
 w10 =[]
 w11 =[]
 w12 =[]
 w13 =[]
 w14 =[]
 w15 =[]

 for i in range(0,len(weights)):
     w1.append(weights[i][0])
 for i in range(0,len(weights)):
     w2.append(weights[i][1])
 for i in range(0,len(weights)):
     w3.append(weights[i][2])
 for i in range(0,len(weights)):
     w4.append(weights[i][3])
 for i in range(0,len(weights)):
     w5.append(weights[i][4])
 for i in range(0,len(weights)):
     w6.append(weights[i][5])
 for i in range(0,len(weights)):
     w7.append(weights[i][6])
 for i in range(0,len(weights)):
     w8.append(weights[i][7])
 for i in range(0,len(weights)):
     w9.append(weights[i][8])
 for i in range(0,len(weights)):
     w10.append(weights[i][9])
 for i in range(0,len(weights)):
     w11.append(weights[i][10])
 for i in range(0,len(weights)):
     w12.append(weights[i][11])
 for i in range(0,len(weights)):
     w13.append(weights[i][12])
 for i in range(0,len(weights)):
     w14.append(weights[i][13])
 for i in range(0,len(weights)):
     w15.append(weights[i][14])

 plt.plot(x,w1)
 plt.plot(x,w2)
 plt.plot(x,w3)
 plt.plot(x,w4)
 plt.plot(x,w5)
 plt.plot(x,w6)
 plt.plot(x,w7)
 plt.plot(x,w8)
 plt.plot(x,w9)
 plt.plot(x,w10)
 plt.plot(x,w11)
 plt.plot(x,w12)
 plt.plot(x,w13)
 plt.plot(x,w14)
 plt.plot(x,w15)
 plt.legend(['SP500','Energy','Precmetandmining','Healthcare','USgrowth','GrowthInt','ValeInt','LtCorpbond','JunkBond','GNMA','STCORPBOND','LTTBOND','PACstock','Eurstock'],loc='upper right',fontsize = 'x-small')
 plt.show()


ppp = plot(time,stock_num,wealth,weights,sp_wealth)

