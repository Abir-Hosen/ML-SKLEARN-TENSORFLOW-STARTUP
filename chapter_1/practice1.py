#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:43:06 2020

@author: abir
"""

from matplotlib.pyplot import plot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


df = pd.read_json('https://bittrex.com/api/v1.1/public/getmarkethistory?market=BTC-ETC')
df = pd.DataFrame(df['result'].values.tolist())
df= df[df["OrderType"]=="SELL"]
df2 = df[["Price", "Quantity", "Total"]]

X1= np.c_[df2["Price"]]
X2= np.c_[df2["Quantity"]]
y= np.c_[df2["Total"]]

df2.plot(kind='scatter', x='Price', y='Total')
df2.plot(kind='scatter', x='Quantity', y='Total')
plt.show()

k_model_for_price = KNeighborsRegressor(n_neighbors=5).fit(X1, y)
k_model_for_quantity = KNeighborsRegressor(n_neighbors=5).fit(X2, y)

lin_reg_for_price = LinearRegression().fit(X1,y)
lin_reg_for_quantity = LinearRegression().fit(X2,y)

X1_Price = [[0.00070252]]
X2_Quantity = [[25]]

k_model_for_price.kneighbors(X1_Price)
k_model_for_quantity.kneighbors(X2_Quantity)

print(k_model_for_price.predict(X1_Price))
print(k_model_for_quantity.predict(X2_Quantity))

print(lin_reg_for_price.predict(X1_Price))
print(lin_reg_for_quantity.predict(X2_Quantity))












