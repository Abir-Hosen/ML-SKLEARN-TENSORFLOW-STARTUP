#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 03:22:04 2020

@author: abir
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

dataframe = pd.read_excel("AIRPOLUTION.xlsx")
dataframe = pd.read_csv("WEO_Data.xls", thousands=',', delimiter='\t', encoding='Latin1', na_values='n/a')
dataframe = pd.read_csv("COVID-19 BD Dataset-5 May.csv")
dataframe = pd.read_csv("heart.csv")

dataframe2 = pd.read_csv("BLI.csv", thousands=",")
dataframe2 = dataframe2[dataframe2["INEQUALITY"]=="HGH"]
dataframe2 = dataframe2.pivot(index="Country", columns="Indicator", values="Value")

dataframe = dataframe[["Country", "2015"]]#for weo gdp per capita
dataframe.rename(columns={"2015":"GDP per capita"}, inplace=True)
dataframe.set_index("Country", inplace=True)

dataframe = dataframe.iloc[list(set(range(59)))]#for bd corona

dframemarge = pd.merge(left=dataframe, right= dataframe2, left_index=True, right_index=True)
dframemargels = dframemarge[["GDP per capita","Life satisfaction"]]
dframemargelsN =  dframemargels.iloc[list(set(range(38)) - set([3,6,7,19,20,23,24,26,28,32,33,35]))]
def Pol(p,q):
    return dframemargelsN.plot(kind="scatter", x=p, y=q, figsize=(10,5), title=p+" VS "+q)

def Polm(p,q):
    return dframemargelsN.plot(kind="hexbin", x=p, y=q, figsize=(10,5), title=p+" VS "+q, grid= True)
#kind: line, bar, barh, hist, box, kde, density, area, pie, scatter, hexbin

Pol("GDP per capita","Life satisfaction")
Polm("GDP per capita","Life satisfaction")
plt.show()


X = np.c_[dframemargelsN["GDP per capita"]]
y = np.c_[dframemargelsN["Life satisfaction"]]

lm = LinearRegression()
lm.fit(X,y)

knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(X,y)

XTest = [[1222],[33333], [55555]]

m = lm.fit(X,y).coef_
c = lm.fit(X,y).intercept_
m
c


yTest = lm.predict(XTest)
yTest = knn.predict(XTest)
yTest
