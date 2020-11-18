#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:21:48 2020

@author: emmanuelgobet
"""

import  numpy as np
import scipy.stats as sps
import sys
import matplotlib.pyplot as plt
import datetime

# Black-Scholes price for Put and Call options

def coeff_d0_d1(Indice, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE):
    if STRIKE<=0.0:
        print(Indice, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)
        sys.exit('Strike is not positive')
    if S0<=0.0:
        print(Indice, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)
        sys.exit('Spot is not positive')
    if VOL**2*MATURITE<=0.0:
        print(Indice, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)
        sys.exit('Total variance is not positive')
    return (np.log(S0 / STRIKE) + (TAUX-DIVIDENDE) * MATURITE) / (VOL * np.sqrt(MATURITE))+ VOL * np.sqrt(MATURITE) * (Indice-0.5)

def price_call_BS(  S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE):
    d0=coeff_d0_d1(0.0, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)
    d1=coeff_d0_d1(1.0, S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)
    return S0 * np.exp(-DIVIDENDE*MATURITE) * sps.norm.cdf(d1) -STRIKE*np.exp(-TAUX*MATURITE)*sps.norm.cdf(d0)

def price_put_BS(S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE):
    return price_call_BS(S0, STRIKE, TAUX, DIVIDENDE, VOL, MATURITE)-(S0 * np.exp(-DIVIDENDE*MATURITE) -STRIKE*np.exp(-TAUX*MATURITE))

# the portefolio to test  (type of optio, which asset, maturity, strike, nominal)
# #MyOptionPortfolio = [['Call', 'Stock0', 1/12, 455.0, 90], ['Put', 'Stock0', 1/12,455.0, 90],
#                      ['Put', 'Stock0', 2/12, 435.0, -130], ['Call', 'Stock0', 2/12, 435.0, -130],
#                      ['Call', 'Stock0', 3/12, 405.0, 50], ['Put', 'Stock0', 3/12, 405.0, 50],
#                      ['Call', 'Stock0', 5/12, 370.0, -12], ['Put', 'Stock0', 5/12, 370.0, -13],
#                      ['Call', 'Stock0', 2/12, 475.0, 72], ['Put', 'Stock0', 2/12, 475.0, 73],
#                      ['Call', 'Stock1', 1/12, 475.0, -50], ['Put', 'Stock1', 1/12, 475.0, -50],
#                      ['Call', 'Stock1', 2/12, 460.0, 50], ['Put', 'Stock1', 2/12, 460.0, 50],
#                      ['Call', 'Stock1',  5/12, 590.0, -15], ['Put', 'Stock1', 5/12, 590.0, -15],
#                      ['Call', 'Stock1', 3/12, 535.0, 12], ['Put', 'Stock1', 3/12, 535.0, 13]]
MyOptionPortfolio = [['Call', 'Stock0', 1/12, 455.0, 1]]
for idx in range(100):
    # model for asset (2 stocks)
    TimeGrid = np.array([0., 10/255, 20/255 ])
    #dates of the problem (10 days, 20 days)

    S0=np.array([440, 460])
    Vol=np.array([0.2, 0.25])
    Correlation=0.3
    InterestRate=5.0
    Drift=np.array([0.01, -0.03])
    Params_Model = {'S0': S0, 'VOL': Vol, 'IR': InterestRate, 'DRIFT': Drift, "CORREL": Correlation}
    class ModelBS_dim2():
        def __init__(self, Params_Model, TimeGrid):
            self.model_id = "Black-Scholes dim2"
            self.S0 = Params_Model["S0"]
            self.vol = Params_Model["VOL"]
            self.cor = Params_Model["CORREL"]
            self.ir = Params_Model["IR"]
            self.drift = Params_Model["DRIFT"]
            self.ndates = len(TimeGrid)
            self.all_time_steps = np.delete(TimeGrid, 0)-np.delete(TimeGrid, -1) # list of time steps
        def sample_path(self):
            path=np.zeros((2,self.ndates))
            Brownian0Path=np.random.normal(loc=0, scale=1, size=self.ndates-1)*np.sqrt(self.all_time_steps)
            Brownian1Path=np.random.normal(loc=0, scale=1, size=self.ndates-1)*np.sqrt(self.all_time_steps)
            path[0,:]=self.S0[0]*np.exp(np.insert(np.cumsum(self.vol[0]*Brownian0Path +(self.drift -0.5*self.vol[0]**2)*self.all_time_steps), 0, 0))
            path[1,:]=self.S0[1]*np.exp(np.insert(np.cumsum(self.vol[1]*(self.cor*Brownian0Path +np.sqrt(1-self.cor**2)*Brownian1Path)+ (self.drift-0.5*self.vol[1]**2)*self.all_time_steps), 0, 0))
            return path

    MyModel = ModelBS_dim2(Params_Model=Params_Model, TimeGrid=TimeGrid)

    # synthetic market data
    NUMBER_MC = 1000
    NUMBER_PLOT = 100 # only for plots
    MarketScenario = np.zeros((NUMBER_MC, 2, len(TimeGrid)))
 # to rerun always the same scenarii (remove later)

    plt.close('all')
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.title("Different market scenarii")
    plt.xlim(0,np.amax(TimeGrid))
    path=MyModel.sample_path()
    for m in np.arange(NUMBER_MC):
        MarketScenario[m, 0, 1] = path[0,1]
    for m in np.arange(NUMBER_MC):
        path=MyModel.sample_path()
        MarketScenario[m, 0, 0] = path[0,0]
        MarketScenario[m, 0, 2] = path[0,2]


    # portfolio simulations
    pricePortfolio = np.zeros((NUMBER_MC, len(TimeGrid)))
    for m in np.arange(NUMBER_MC):
        for t in np.arange(len(TimeGrid)):
            for option in MyOptionPortfolio:
                OptionId=option[0]
                StockId=option[1]
                Maturity=option[2]
                Strike=option[3]
                Number=option[4]
                if OptionId=="Call":
                    if StockId=="Stock0":
                        pricePortfolio[m, t] += Number*price_call_BS(MarketScenario[m, 0, t], Strike, MyModel.ir,
                                                            0, MyModel.vol[0], Maturity-TimeGrid[t])
                    else: # this is stock1
                        pricePortfolio[m, t] += Number*price_call_BS(MarketScenario[m, 1, t], Strike, MyModel.ir,
                                                            0, MyModel.vol[1], Maturity-TimeGrid[t])
                else: # this is a put
                    if StockId=="Stock0":
                        pricePortfolio[m, t] += Number*price_put_BS(MarketScenario[m, 0, t], Strike, MyModel.ir,
                                                            0, MyModel.vol[0], Maturity-TimeGrid[t])
                    else: # this is stock1
                        pricePortfolio[m, t] += Number*price_put_BS(MarketScenario[m, 1, t], Strike, MyModel.ir,
                                                            0, MyModel.vol[1], Maturity-TimeGrid[t])



    #plt.show()

    # putting data in csv (stock0 at time 1 ( after 10 days ), stock1 at time 1, price at time 1-price at time 2 (after 20 days) =loss)
    SyntheticData=np.zeros((NUMBER_MC,2))
    for m in np.arange(NUMBER_MC):
        SyntheticData[m,0]=MarketScenario[m, 0, 1]
    # SyntheticData[m,1]=MarketScenario[m, 1, 1]
        SyntheticData[m,1]=pricePortfolio[m, 1]-pricePortfolio[m, 2]

    #np.savetxt('SyntheticData'+str(datetime.datetime.now())+'.csv', SyntheticData, delimiter=',', fmt='%f', header='stock0@time1,stock1@time1,portoliolossBetweentime1and2')

    np.savetxt('C:/Users/malex/Desktop/scrm/code/tmpdata/SyntheticData'+str(idx)+'.csv', SyntheticData, delimiter=',', fmt='%f', header='stock0@time1,portoliolossBetweentime1and2')
