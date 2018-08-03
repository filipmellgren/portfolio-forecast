#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:13:25 2018

@author: filip
"""
import numpy as np
import matplotlib.pyplot as plt
# Define weight on risky asset
n_portfolios = 1
portfolio_weights = np.linspace(0,1,n_portfolios)

# Assume the following rates of return for the risky asset and risk free asset.
rf_return = 0.02
M_return = rf_return + 1*(0.07 - rf_return)

# Calculate portfolio expected returns
portfolio_returns = portfolio_weights*rf_return + (1-portfolio_weights)*M_return

# Draw out a time dimension
years = 10
time = np.linspace(0, years, years*12)

# Calculate mean trajectory for the differently weighted portfolios over time. 
time.shape = (len(time), 1)
portfolio_returns.shape = (1, len(portfolio_returns))
portfolios = np.exp(time@portfolio_returns)

# Calculate volatilities
# Volatility random walk: https://stats.stackexchange.com/questions/159650/why-does-the-variance-of-the-random-walk-increase
sigma_M = 0.15 # Annual volatility for the risky asset "M". Apx VIX levels.
portfolio_vol = (1-portfolio_weights)*np.sqrt(time)*sigma_M # Time dependent
portfolio_95 = portfolio_vol*(1.97) # 1.97 for 95% CI (assuming normal dist)

# Create a plot for the figures:
with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots()
    ax.plot(time, portfolios)
    ax.legend(portfolio_weights.round(2), loc = "upper left", 
              title = "Risk free asset share ")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Portfolio development with 95% CI")

# Add 95 percent CI. We deal with a non stationary stochastic process and 
# assume that volatility increases proportionally with the level of the series.
# This means we have to make a logarithmic transformation of the series
# (to achieve constant variance) and then exponentiate again. (Box-Cox transf.)
    for p in range(n_portfolios):
        plt.fill_between(time.flatten(),
                     np.exp(np.log(portfolios[:,p]) - portfolio_95[:,p]), 
                     np.exp(np.log(portfolios[:,p]) + portfolio_95[:,p]),
                     alpha = 0.5)
        
        
        
# Simulate random data to check if CIs look correct
n_simulations = 100
random_returns = np.ones((years*12, n_simulations))
for s in range(n_simulations):
    random_returns[:,s] = np.random.normal(1 + M_return/12, 
                                  sigma_M/np.sqrt(12), years*12)
    
random_trajectories = np.cumprod(random_returns,0)

with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots()
    ax.plot(time, portfolios)
    ax.plot(time, random_trajectories, linewidth = 0.5)

    ax.legend(portfolio_weights.round(2), loc = "upper left", 
              title = "Risk free asset share ")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Portfolio development with 95% CI")
    #plt.ylim(ymax = 10, ymin = 0)

    for p in range(n_portfolios):
        plt.fill_between(time.flatten(),
                     np.exp(np.log(portfolios[:,p]) - portfolio_95[:,p]), 
                     np.exp(np.log(portfolios[:,p]) + portfolio_95[:,p]),
                     alpha = 0.5)




