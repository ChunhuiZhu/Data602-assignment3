# -*- coding: utf-8 -*-
"""
Created on Sun April  8 18:30:05 2018

@author: Chunhui Zhu
"""
import pandas as pd
import requests
import urllib.request as web
import json 
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import datetime as DT
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
from dateutil.parser import parse
import itertools
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
reg=LinearRegression()




#checkticker(symbol)is for checking if symbol is correctly inputed by trader
#Aftet user entry the symbol, lookup its company name form coinmarketcap.com
#return company name which will use in history100day(com)
def checkticker(symbol):
    url="https://api.coinmarketcap.com/v1/ticker/"
    jdata=requests.get(url).json()
    for i in jdata:
        if i["symbol"]==symbol:
            company=i["name"]
            return(company)    


#updatedprice(symbol) will excutive after the trade confirmed
#return an array[ask, bid] to Trade() for order calculation 
def updatedprice(symbol):
        ticker="USDT-"+symbol
        url="https://bittrex.com/api/v1.1/public/getorderbook?market="+ticker+"&type=both"
        jdata=requests.get(url).json()
        #"sell" in API reflex to ask price
        data=jdata["result"]["sell"][0]
        askprice=data['Rate']
        
        #"buy" in API reflex to bid price
        data=jdata["result"]["buy"][0]
        bidprice=data['Rate']
        
        newprice=[askprice,bidprice]
        return(newprice)

    

#showtrade() will be excutive before the trader confirms the trade
#last 24hr trading data is from bittrex.com
#show basic analytics: average price, max price, min price, variance of price
def showtrade(symbol):
    ticker="USDT-"+symbol
    url="https://bittrex.com/api/v1.1/public/getorderbook?market="+ticker+"&type=both"
    jdata=requests.get(url).json()
    
    #"sell" in API reflex to ask price
    buydata=jdata["result"]["sell"]
    dfbuy=pd.DataFrame.from_dict(buydata)
    meanbuy=dfbuy["Rate"].mean()
    maxbuy=max(dfbuy["Rate"])
    minbuy=min(dfbuy["Rate"])
    stdbuy=np.std(dfbuy["Rate"])
    
    #"buy" in API reflex to bid price
    selldata=jdata["result"]["buy"]
    dfsell=pd.DataFrame.from_dict(selldata)
    meansell=dfsell["Rate"].mean()
    maxsell=max(dfsell["Rate"])
    minsell=min(dfsell["Rate"])
    stdsell=np.std(dfsell["Rate"])
    
    print("")
    print("----------------------------------------------------------------")
    print(" Bittrex.com: USDT-"+symbol+"       Last 24hr Price: buy/sell   ")
    print("----------------------------------------------------------------")
    print("     Price             Buy                     Sell             ")
    print("----------------------------------------------------------------")
    print("   Average:      "+"{0:.10f}".format(meanbuy)+"         "+"{0:.10f}".format(meansell))
    print("   Max    :      "+"{0:.10f}".format(maxbuy)+"         "+"{0:.10f}".format(maxsell))
    print("   Min    :      "+"{0:.10f}".format(minbuy)+"         "+"{0:.10f}".format(minsell))
    print("   Std    :      "+"{0:.10f}".format(stdbuy)+"           "+"{0:.10f}".format(stdsell))
    print("----------------------------------------------------------------")
    print("")




def historyprice(com,numdays):   
    #use datatime function to calculate the date before 100 days
    #and store as backdate base on the designed formate
    today=DT.date.today()
    backdate= today-DT.timedelta(days=numdays)
    formate='%Y%m%d'
    backdate=backdate.strftime(formate)
    today=today.strftime(formate)

    #assign today and back100date to url date range
    #https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20180305&end=20180404
    url="https://coinmarketcap.com/currencies/"+com+"/historical-data/?start="+backdate+"&end="+today
    soup = BeautifulSoup(requests.get(url, "lxml").content)
    headings=[th.get_text() for th in soup.find("tr").find_all("th")]
    
    histdata=[]
    for row in soup.find_all("tr")[1:]:
        rowdata = list(td.get_text().replace(",","") for td in row.find_all("td"))
        histdata.append(rowdata)   

    #stor histroy data in a panda df his100
    #Change the type of datetime and order data by date
    hist=pd.DataFrame(histdata,columns=headings)
    hist=hist.convert_objects(convert_numeric=True)
    
    hist['Date'] = [parse(d).strftime('%Y-%m-%d') for d in hist['Date']]
    hist=hist.sort_values(by='Date')

    return(hist)


#his100chart(company) will excutive after checking trader input is correct 
#get data by calling history100day(com)
#perform data visulatizations for last 100 days trading data
def his100chart(company):
    company=company.lower()
    df100=historyprice(company,100)

    #show 20 day moving averages
    print("Last 100-day trade history chart")
    
    df100["CloseMean"]=df100["Close"].rolling(20).mean()
    df100.reset_index(inplace=True)
    df100.index = df100.Date

    
    fig, ax = plt.subplots()
    plt.xticks(rotation = 90)
    plt.xlabel("Last 100 days")
    plt.ylabel("USD Price")
    plt.title(company+"/USD : 20 day moving averages" )
    
    candlestick2_ohlc(ax, df100["Open"], df100["High"], df100["Low"], df100["Close"], width=1, colorup='g')
    df100.CloseMean.plot(ax=ax)
    plt.grid()
    plt.show()
    
 
    




#Trade(amount) function excutive following steps of trade:
#checkticker(symbol) will check if input can be found in the trade in coinmarketcap.com
#his100chart(company) show 100-day datachart 
#showtrade(symbol) show basic analytics for decision making 
#if trade decides to trade and enter number of the trade
#updatedprice(symbol) return instant bid/askprice after confirm number of the trade
#pfoilodf,vwapdf are for account portfolio(cash+assets) and vwap price(per asset) time series chart, which update the recorde when Trade() be selected 
def Trade(histdf,pldf,amount,pfoilodf,vwapdf):
    company=None
    while not company:
        symbol=str(input("Enter a ticker (exmaple: BTC) : "))
        symbol=symbol.upper()
        company=checkticker(symbol)
        if not company:
            print("Ticker not found. ")
            

    his100chart(company)
    showtrade(symbol)
    
    decision=int(input("Do you want to trade (Enter 1:Yes, 0:No.)?  :  ")) 
    if decision==1:
        #The user is then asked to confirm the trade at the market price by enter the quantity.
            while True:
                try:
                     quantity=float(input("Enter number of share(s) (positive for buy/ negative for sell): ")) 
                     break
                except ValueError:
                     print("Wrong input! Try again.")
                     
            if quantity !=0.000000000000 :  
                newprice=updatedprice(symbol)
                if quantity>0.000000000000:
                    cost=newprice[0]*quantity
                    
                    #check if cash account has enought money to buy
                    while(amount<cost):
                        print("Your Account: " + str(amount)+ "   |     Cost : " +str(cost))
                        print("You don't have enought money in your account. ")
                        quantity=float(input("Enter number of share(s) (positive for buy/ negative for sell / 0 back to manu): ")) 
                        #get instant price
                        newprice=updatedprice(symbol)
                        cost=newprice[0]*quantity
                    
                  
                    amount-=cost 
                    order={"Company":company,"Symbol":symbol,"Side":"buy","Volumn":quantity,"Price":newprice[0],"Total Cost":cost,"Time":time.strftime("%c"), "Cash Account" : amount}
                    print("Total cost :  ",cost)
                    print("Cash Account : ",amount)
                
                    histdf=histdf.append(order,ignore_index=True)
                    pldf,vwapdf=updatePL(pldf,order,vwapdf) 

                          
                #if quantity is negative, recorde add as sell stock with bid price
                elif quantity<0.0000000000000 :
                    cost= newprice[1] *quantity
                    amount-=cost 
                    order={"Company":company,"Symbol":symbol,"Side":"sell","Volumn":quantity,"Price":newprice[1],"Total Cost":cost,"Time":time.strftime("%c"), "Cash Account" : amount}
                    print("Total cost :  "+ str(cost))  
                    print("Cash Account : ",amount)
                    
                    histdf=histdf.append(order,ignore_index=True)  
                    pldf,vwapdf=updatePL(pldf,order,vwapdf) 
    
                #update account portfolio at the time when user get into trade /willing to trade 
                #recorde the time point at the same time
                #create a series data for total_portfolio chart
                total_portfolio= update_pfoliodf(pldf,amount)
                pfoilodf = pfoilodf.append({'Time': order['Time'],'Total Portfolio': total_portfolio}, ignore_index=True) 
            return(histdf,pldf,amount,pfoilodf,vwapdf)
    else:
            return(histdf,pldf,amount,pfoilodf,vwapdf)


#when new trade executive, call updataPL(), calculate the profit and loss and update cash amount in account
#each ticker only has maximun 1 recorde in the table  
#Rpl updated only when coins are sold
#Upl is not change
#calculate vwap for each assets and apply to time series chart
                  
def updatePL(pldf,order,vwapdf):
    
    old=pd.DataFrame()
   
    #order=df["Company","Symbol","Side","Volumn","Price","Total Cost","Time", "Cash Position"]
    #plcolname=["Symbol", "Inventory","Wap","Rpl","Upl","Time"]
    if pldf.empty:
        new={"Symbol":order["Symbol"],"Inventory":order["Volumn"],"Wap":order["Price"],"Rpl":0.00,"Upl":0.00,"Time":order["Time"]} 
        pldf=pldf.append(new,ignore_index=True) 
        vwapdf = vwapdf.append({"Symbol":order["Symbol"],'Wap': order["Price"],'Time': order["Time"]}, ignore_index=True) 
    else:     
        old=pldf.loc[pldf["Symbol"]==order["Symbol"]]
        if not old.empty:
            print("")
            print("rwtretwetregfsdgsdfbfsdgjfdpgifajdpigjapgjfdpj")
            
            #If find a recorde, then drop it from the pldf
            pldf= pldf.drop(pldf[pldf["Symbol"]==order["Symbol"]].index)
    
            if order["Volumn"]<0.00000 and old["Inventory"][0]>0.00000:
                Rpl=(order["Price"]-old["Wap"][0])*min(abs(order["Volumn"]),abs(old["Inventory"][0]))
            
            #if it is short position, buying use ask price 
            elif order["Volumn"]>0.00000 and old["Inventory"][0]<0.00000 :
                Rpl=(old["Wap"][0]-order["Price"])*min(abs(order["Volumn"]),abs(old["Inventory"][0]))    
            
            else:
                Rpl=0.00000
         
            #For wap (is absolute positive representing a signle price of share of buy and sell:
            inven=old["Inventory"][0]+order["Volumn"]
            if( inven!=0.000000):
                Wap=(old["Wap"][0]*old["Inventory"][0]+order["Price"]*order["Volumn"])/inven    
            else:
                Wap=0.00000
            
            new={"Symbol":order["Symbol"], "Inventory":inven,"Wap":Wap,"Rpl":Rpl,"Upl":0.00000,"Time":order["Time"]}
            pldf=pldf.append(new,ignore_index=True) 
            
            vwapdf = vwapdf.append({"Symbol":order["Symbol"],'Wap': Wap,'Time': order["Time"]}, ignore_index=True) 
        else:
             new={"Symbol":order["Symbol"],"Inventory":order["Volumn"],"Wap":order["Price"],"Rpl":0.00000,"Upl":0.00000,"Time":order["Time"]} 
             pldf=pldf.append(new,ignore_index=True) 
             vwapdf = vwapdf.append({"Symbol":order["Symbol"],'Wap': order["Price"],'Time': order["Time"]}, ignore_index=True) 
    

    print(" end   updatePL")
    print(pldf)
    print(order)
    print(vwapdf)
    
    return(pldf,vwapdf)


#histlistcharts will show time series data for cash position, portfolio, VWAP, history price for crypo
#def histlistcharts(histdf):
    




#showPL() undated only when user selction Show P/L .      
#showPL() is to update the unreal profit/loss using updated ask/bid price to est.
#Upl is only data will be changed regarding to exist ticker.
#updateprice() has (tick,askprice,bidprice,time.strftime("%c"))
#pldf has items as in pl_col=["Symbol", "Inventory","Wap","Rpl","Upl","Time"]
#inventory is negative, to cover short sell, upl use ask price; else use bid price.
def showPL(pldf):
    
    if pldf.empty:
        return (pldf)
    else:
        for i in range(len(pldf)): 
            newprice=updatedprice(pldf.loc[i,"Symbol"]) 
           
            if pldf.loc[i,"Inventory"]>0.00 :
                pldf.loc[i,"Upl"]=(newprice[1]-pldf.loc[i,"Wap"])*pldf.loc[i,"Inventory"]
                
            #For short position, upl use ask price; else upl use ask price  
            elif pldf[i,"Inventory"]<0.00 :
                pldf.loc[i,"Upl"]=(pldf.loc[i,"Wap"]-newprice[0])*pldf.loc[i,"Inventory"]
               
            else: 
                pldf.loc[i,"Upl"]=0.0
                pldf.loc[i,"Wap"]=0.0
    return(pldf)  
    




#pfolio() use Markov Chain Monte Carlo simulation methods
#Get the best two weighted protfolios in assets
#Sharp-ratio : risk-free rate use treasure bone yeild rate convert to compond daily return
#further analysis purpose: breakdow in training and testing setsevaluate prediction accuracy
def pfolio():
        assets=[]
        print("Suggest at least 3 assets in your portfolio, example: bitcoin,ethereum,ripple.")
        numassets=int(input("Enter number of assets in your porfolio (int) : "))
        
        for i in range(0,numassets): 
            a=str(input("Asset "+str(i)+ " : "))
            assets.append(a)
        ndays_return=int(input("Enter # of day's return in your porfolio (int) : "))    
        
        #create the df (histprice) to collecting close price of 365 days history data
        histprice=pd.DataFrame({'A' : []})
        for i in assets:
                a=historyprice(i,730)['Close']  
                if histprice.empty:
                    histprice= a
                else:
                    histprice=pd.concat([histprice, a], axis=1)
        
        histprice=histprice.convert_objects(convert_numeric=True)
        
        print("")
        
        #1.portfolio weights and returns
        #log(dailyreturn) as log_dr
        log_dr=np.log(histprice/histprice.shift(ndays_return))
        log_dr=log_dr.dropna()
        log_dr.columns = assets
        #print(log_dr)
        
    
        log_dr.plot()
        plt.ylabel('Daily Return')
        plt.title('2 yrs History Daily Retrun')
        plt.grid()
        plt.show()
        
        #2.further analysis purpose: breakdow in training and testing sets
        #train_log=log_dr[0:int(0.8*len(log_dr))]
        #test_log=log_dr[int(0.8*len(log_dr)):]
        train_log=log_dr[0:int(1*len(log_dr))]
        
        num_assets=len(assets)
        
        #follwing are collecting data for graph and analysis
        pfolio_returns = []
        pfolio_volatilities = []
        sharpe_ratio = []
        coin_weights = []
        
        np.random.seed(101)
        
        #3.drivers of performance
        #simulation 20000 different portfolio to calculate the result
        for x in range (20000):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.sum(weights * train_log.mean())
            volatility=np.sqrt(np.dot(weights.T, np.dot(train_log.cov(),weights)))
            
            pfolio_returns.append(returns)
            pfolio_volatilities.append(volatility)
            #Assume risk free intreste is 0.000075 compond daily return in sharpe_ratio
            sharpe_ratio.append( (returns-0.000075*ndays_return) / volatility)
            coin_weights.append(weights)
            
        #built portfolios df for all collecting simulated data    
        pfolio_returns=np.array(pfolio_returns)
        pfolio_volatilities=np.array(pfolio_volatilities)
        pfolios = pd.DataFrame({'Return' : pfolio_returns, 'Volatility' : pfolio_volatilities, 'Sharpe_ratio' : sharpe_ratio})
        coinw=pd.DataFrame(coin_weights,columns=assets)
        
        portfolios = coinw.join(pfolios)
        
        
        #4.portfolio optimization
        # find min Volatility & max sharpe values in the dataframe (portfolios)
        min_volatility = portfolios['Volatility'].min()
        max_sharpe = portfolios['Sharpe_ratio'].max()
        
        # use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = portfolios.loc[portfolios['Sharpe_ratio'] == max_sharpe]
        min_variance_port = portfolios.loc[portfolios['Volatility'] == min_volatility]
        
    
        print(" ")
        #Graph 20000 portfolios 
        plt.style.use('seaborn-dark')
        portfolios.plot.scatter(x='Volatility', y='Return', c='Sharpe_ratio',
                        cmap='RdYlGn', edgecolors='black', figsize=(8, 6), grid=True)
        plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Return'], c='red', marker='D', s=100)
        plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Return'], c='blue', marker='D', s=100 )
        plt.xlabel('Expected '+str(ndays_return)+'-day Volatility')
        plt.ylabel('Expected '+str(ndays_return)+'-day Returns')
        plt.title('Efficient '+str(ndays_return)+'-day Frontier')
        plt.show()
        
        print("Risk free intreste based on 0.0075% compond daily")
        print("")
        # print the details of the 2 special portfolios
        print("Result based on 20,000 portfolios")
        print("")
        print("Min variance Portfolios (blue dot): ")            
        print(min_variance_port.T )            
        print("")
        print("")
        print("Max Sharpe Ratio Portfolios (red dot):")
        print(sharpe_portfolio.T)
        
               
        
#Use SARIMAX to predict price   
def best_SARIMAX(hist730, num_param):
    # Define the p, d and q parameters to take any value between 0 and num_param-1
    p = d = q = range(0, num_param)
    P = D = Q = range(0, num_param)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    PDQ = list(itertools.product(P, D, Q))
    del pdq[0] #remove (0,0,0)
    del PDQ[0] #remove (0,0,0)
    
    #use AIC to eveluate model performence
    best_aic=100000000000
    
    best_pdq = pdq[1]
    best_PDQ = PDQ[1]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    for i in pdq:
        for j in PDQ:
            try:
                mod = sm.tsa.statespace.SARIMAX(hist730, trend='n', order=i, seasonal_order=(j[0],j[1],j[2],12))
                results = mod.fit()
                p=results.pvalues
                
                if all(k < 0.05 for k in p): 
                    if (results.aic < best_aic):
                        best_aic = results.aic
                        best_pdq = i
                        best_PDQ = j 
            except:
                 continue
                
    return (best_pdq,best_PDQ)

    
#Use regression model to predict price
#further analysis in this part :  breakdow in training and testing sets, evaluate prediction accuracy
def regression(hist,assets,ndays):
    histprice =hist.dropna()

    histprice=histprice.convert_objects(convert_numeric=True)
    
    for i in range(len(assets)):          
        x = histprice.drop(histprice.iloc[:,i-1] , axis=1)
        y = histprice.iloc[:,i-1] 
    
        #Note the difference in argument order
        model = reg.fit(x,y)
        #print(model)
        predictions = model.predict(x) # make the predictions by the model
        print(assets[i-1] + " regression prediction price :  " + str(predictions[-1]) + "  in " + str(ndays)+" days .")
   
    
    
#pred() calls two functions: regression() and best_SARIMAX()
#Use  skilearn and statsmodels buided-in functions to predict next ndays' price
def pred():
    #"bitcoin","ripple","ethereum","litecoin"
    assets=[]
    histprice=pd.DataFrame({'A' : []})
    
    print("Suggest at least 2 assets in your prediction,example: bitcoin,zcash,ethereum,ripple.")
    numassets=int(input("Enter number of assets: "))
    for i in range(0,numassets): 
        a=str(input("Asset "+str(i)+ " : "))
        assets.append(a)
    
    ndays=int(input("Enter number of day's for the price prediction : "))    
    
    for k in assets:
            #use 2 yrs daily price 
            hist730=historyprice(k,730)
            hist730.reset_index(inplace=True)
            hist730.index = hist730.Date
            hist=hist730.Close
            
            #prepare a df for regression model
            if histprice.empty:
                histprice= hist
            else:
                histprice=pd.concat([histprice, hist], axis=1)
            hist=hist.dropna()
            
            #get parameters for SARIMAX model
            a,b=best_SARIMAX(hist, 2)
            #a=(1,0,1), b=(1,1,1)
            sarimax_model= sm.tsa.statespace.SARIMAX(hist, trend='n', order=a, seasonal_order=(b[0],b[1],b[2],12)).fit()
            #print(sarimax_model.summary())
        
            yesterday=DT.date.today()-DT.timedelta(days=1)
            after7date= DT.date.today()+DT.timedelta(days=ndays)
            y=yesterday.strftime('%Y-%m-%d')
            f=after7date.strftime('%Y-%m-%d')
        
            #sarimax_model.plot_diagnostics(figsize=(8, 6))
            #plt.show()
        
            pred_out = sarimax_model.get_prediction(start = y, end = f, dynamic = False, full_results = True)
            pred_means = pred_out.predicted_mean
            price=pred_means.iloc[-1]
            print(k + " SARIMAX prediction price :  " + str(price) + "  in " + str(ndays)+" days .")
              
    regression(histprice,assets,ndays)
    
    return
    




#hist_col=["Company","Symbol","Side","Volumn","Price","Total Cost","Time","Cash Account" ]
#pfoilodf{'Time','Total Portfolio'}
#vwapdf{"Symbol",'Wap','Time'}
def pfoilio_chart(histdf, pfoilodf, vwapdf):

    print("")
    #Time-series chart for cash position in the P/L 
    print("--------------------------------------")
    print("            Cash Account              ")
    print("--------------------------------------")
    plt.scatter(x=histdf['Time'], y=histdf["Cash Account"] , c='r', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.title("Cash Account")
    plt.grid(True)
    plt.show()


    print("")
    #Time-series chart for  total portfolio in the P/L
    print("--------------------------------------")
    print("             Total Portfolio          ")
    print("--------------------------------------")
    plt.scatter(x=pfoilodf['Time'], y=pfoilodf['Total Portfolio'], c='r', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Portfolio')
    plt.title("Total Portfolio")
    plt.grid(True)
    plt.show()
 
    
    print("")
    Symbol = histdf.Symbol.unique()
    #Time-series chart for VWAP of each Symbol 
    print("--------------------------------------")
    print("            VWap by Symbol            ")
    print("--------------------------------------")
    for i in Symbol:
        plt.scatter(x=vwapdf.loc[vwapdf["Symbol"] == i]['Time'], y=vwapdf.loc[vwapdf["Symbol"] == i]['Wap'], c='r', alpha=0.5)
        plt.title(i)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
 
                
    print("")
    #4.d Time-series chart for executed order prices by Symbol
    print("--------------------------------------")
    print("    Executed Trade Prices by Tickers  ")
    print("--------------------------------------")
    for i in Symbol:
        plt.scatter(x=histdf.loc[histdf["Symbol"] == i]['Time'], y=histdf.loc[histdf["Symbol"] == i]['Price'], c='b', alpha=0.5)
        plt.title(i)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
 



#update_pfoliodf() to calculate the amount assets(Symbol) and cash (amount) at the time momnet been called
#pl_col=["Symbol", "Inventory","Wap","Rpl","Upl","Time"]
def update_pfoliodf(pldf,amount):
    rows = pldf.shape[0]
    for i in range(rows):
        ask,bid= updatedprice(list(pldf['Symbol'])[i])        
   
        if(pldf.Inventory[i] >= 0):
            pldf.Upl[i] = round((pldf.Inventory[i] * (pldf.Wap[i] - bid)),2)
        elif(pldf.Inventory[i] < 0):
            pldf.Upl[i] = round((pldf.Inventory[i] * (pldf.Wap[i] - ask)),2)
            
        amount = amount + pldf.Upl[i]
        
    return(amount)




#connect with mongodb at the begining, 
#if there are datarows in pl and account in mongodb, 
#recorde pl and account_balance to local variables,  
#drop the pl and account_balance in mongodb. 
#new pl and account_balance will reinstore in mongodb at the end
if __name__=="__main__":
    #creat empty df and list to storage current trade data from Mongodb
    histdf=pd.DataFrame()
    pldf=pd.DataFrame()
    
    #df for total portfolio time series chart
    pfoilodf=pd.DataFrame()
    #df for vwap time series chart
    vwapdf=pd.DataFrame()
 
    cl = MongoClient('localhost', 27017)
    
    """
    PLMongo.drop()
    AcctMongo.drop()
    HistoryMongo.drop()
    """

    HistoryMongo= cl["local"]["history"]
    if HistoryMongo.count() !=0:
        histdf=pd.DataFrame(list(HistoryMongo.find()))
        del histdf[list(histdf)[-1]]
        HistoryMongo.drop()
    else:    
        hist_col=["Company","Symbol","Side","Volumn","Price","Total Cost","Time","Cash Account" ]
        histdf = pd.DataFrame(columns = hist_col)    
    

    VwapMongo= cl["local"]["vwap"]   
    if VwapMongo.count() !=0:
        vwapdf=pd.DataFrame(list(VwapMongo.find()))
        del vwapdf[list(vwapdf)[-1]]
        VwapMongo.drop()
    else:        
        vwapdf_col=["Symbol","Time","Wap"]
        vwapdf = pd.DataFrame(columns = vwapdf_col)  
        
        
    pfoiloMongo= cl["local"]["portfolio"]   
    if pfoiloMongo.count() !=0:
        pfoilodf=pd.DataFrame(list(pfoiloMongo.find()))
        del pfoilodf[list(pfoilodf)[-1]]
        pfoiloMongo.drop()
    else:        
        pfoilo_col=["Time","Total Portfolio"]
        pfoilodf= pd.DataFrame(columns = pfoilo_col)      
    
    PLMongo= cl["local"]["pl"]
    if PLMongo.count() !=0:
        pldf=pd.DataFrame(list(PLMongo.find()))
        del pldf[list(pldf)[-1]]
        PLMongo.drop()
    else: 
        pl_col=["Symbol", "Inventory","Wap","Rpl","Upl","Time"]
        pldf = pd.DataFrame(columns = pl_col)  
 
    
    AcctMongo= cl["local"]["account"]  #only for recorde the currenct balance 
    if AcctMongo.count() == 0:
        amount=1000000.00
    else:
        account=list(AcctMongo.find()) 
        amount=account[0]["Account_Balance"]
        AcctMongo.drop()
        
   



     
    print("Your cash account : ",amount)
    
    option=1 
    
    while option != 7 : 
            print("")
            print("===========================================")
            print("                 Menu                      ")
            print("===========================================")
            print("             1.  Trade                     ")
            print("             2.  Show Blotter              ")
            print("             3.  Show P/L                  ")
            print("             4.  Investment Portfolio      ")
            print("             5.  Price Prediction          ")
            print("             6.  Quit                      ")
            print("===========================================")
          

            while True:
                try:
                     option = int(input("Please enter a number (1-6): "))
                     break
                except ValueError:
                     print("Wrong input! Try again.")
                   
            while(option<1 or option >6) :
                print("Wrong number! Try again.")
                print("\n")
                option = int(input("Please enter a number (1-6): ")) 
            
            #Start to trade
            if (option==1): 
                histdf,pldf,amount,pfoilodf,vwapdf=Trade(histdf,pldf,amount,pfoilodf,vwapdf)
                print("_________________________________________________")
          

            #Displays the trade blotter, a list of historic trades made by the user. The trade blotter will display
            #the following trade data, with the most recent trade at the top
            elif (option==2):                
                if histdf.empty:
                    print("")
                    print("No trading record.")
                    print("")
                else:    
                    print("")
                    print("======================================")
                    print("            Trading History           ")
                    print("======================================")          
                  
                    print(histdf[::-1]) 
 
                print("_________________________________________________")


            #Displays the profit / loss statement. The P/L will display, 
            #the following trade data, with the most recent trade at the top 
            #Ticker, Position, Current Market Price, VWAP, UPL (Unrealized P/L), RPL (Realized P/L)
            
            elif (option==3):
            
                if pldf.empty:
                    print("")
                    print("Your do not have trades in your account.")
                    print("")

                else:
                    pldf=showPL(pldf)
                    print("")
                    print("======================================")
                    print("              Profit/Loss             ")
                    print("======================================")           
                    #arrange the column order based on plcolname
                    #pldf=pldf[["Symbol", "Inventory","Wap","Rpl","Upl","Time"]]
                    print(pldf[::-1])  
                    
                pfoilio_chart(histdf, pfoilodf, vwapdf)
                print("_________________________________________________")   
                
                
            elif (option==4):
                    print("")
                    print("======================================")
                    print("          Portfolio Evaluation        ")
                    print("======================================")
                    print(" ")
                    pfolio()
             
                
            elif (option==5):
                    print("")
                    print("======================================")
                    print("            Price Prediction          ")
                    print("======================================")
                    print(" ")
                    pred()
             
              
                    
            #Quit when option==6
            else: 
                AcctMongo.insert({"Account_Balance":amount})
                if not histdf.empty:
                    #use ".to_dict('records')" method to convert df to dict
                    HistoryMongo.insert_many(histdf.to_dict('records'))
                if not pldf.empty:         
                    PLMongo.insert_many(pldf.to_dict('records'))
                if not vwapdf.empty:
                    VwapMongo.insert_many(vwapdf.to_dict('records'))               
                if not pfoilodf.empty:         
                    pfoiloMongo.insert_many(pfoilodf.to_dict('records'))
                cl.close()
                print("Good Luck!")
                print("_________________________________________________")
                break;





















