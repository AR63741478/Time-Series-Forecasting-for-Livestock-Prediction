from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *

import warnings
warnings.filterwarnings("ignore")

from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, Theta, OptimizedTheta, AutoETS

pd.set_option('display.max_rows', None)
i = 1
df = pd.read_csv("/Users/arhaann/Downloads/Slaughtered.csv", parse_dates=["Year"])
df = df.rename(columns={'Year': 'ds', 'Entity': 'unique_id','Cattle (cattle slaughtered)': 'Cattle', 'Goats (goats slaughtered)': 'Goat', 'Chicken (chicken slaughtered)': 'Chicken', 'Turkey (turkeys slaughtered)': 'Turkey', 'Pigs (pigs slaughtered)': 'Pig', 'Sheep (sheeps slaughtered)': 'Sheep'})
df = df.drop('Code', axis=1)

afghan = df[df['unique_id'] == 'Afghanistan'].copy()

nai = 0
ha = 0
t = 0
ot = 0
aets = 0

def show_graph():

    Country = str(input("Enter a country with the first letter being capitalized: "))
    Livestock = str(input("Enter a type of livestock from the following options(Cattle, Goats, Chicken, Turkey, Pigs, Sheep): "))

    filter_data = df[df['unique_id'] == Country]


    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filter_data, x='ds', y=Livestock)
    plt.title(f'{Livestock} Slaughtered in {Country} Over Time')
    plt.xlabel('Year')
    plt.ylabel(f'Number of {Livestock}')
    plt.show()


afghan = df.drop(["Pig", 'Turkey'], axis=1)
afghan = afghan[['unique_id', 'ds', 'Cattle']].copy()  #Cattle
afghan = afghan.rename(columns={'Cattle': 'y'})
afghan = afghan.dropna()
afghan = afghan.iloc[:49, :]


horizon = 1
models = [Naive(), HistoricAverage(), Theta(season_length=1), OptimizedTheta(season_length=1), AutoETS(season_length=1)]
sf = StatsForecast(models=models, freq="Y")
sf.fit(df=afghan)
pred = sf.predict(h=horizon)
pred = pred.reset_index()
while i<len(afghan):
    nai = abs(df.iloc[i, 4] - pred["Naive"][0]) + nai
    ha = abs(df.iloc[i, 4] - pred["HistoricAverage"][0]) + ha
    t = abs(df.iloc[i, 4] - pred["Theta"][0]) + t
    ot = abs(df.iloc[i, 4] - pred["OptimizedTheta"][0]) + ot
    aets = abs(df.iloc[i, 4] - pred["AutoETS"][0]) + aets
    i = i+1

errors = {
    'Naive': nai,
    'HistoricAverage': ha,
    'Theta': t,
    'OptimizedTheta': ot,
    'AutoETS': aets
}
smallest_name = min(errors, key=errors.get)
print(smallest_name)

def predict():
    n = int(input("How many years would you like to predict(Ex. 10 = 10 years into the future): "))
    country = input(("Enter the country you would like to forcast(First letter must be capitilized and the country must have correct spelling): "))
    livestock_type = input(("Enter livestock type(First letter capitilized and singular)): "))
    model = OptimizedTheta(season_length = n)
    sf = StatsForecast(models=[model], freq="Y")
    data = df[df['unique_id'] == livestock_type].copy()

    c = df[df['unique_id'] == country].copy()
    c = c[['unique_id', 'ds', livestock_type]].copy()  #Cattle
    c = c.rename(columns={livestock_type: 'y'})

    sf.fit(df=c)
    predict = sf.predict(h=n)
    predict = predict.reset_index()
    predict = predict.rename(columns={'unique_id': 'Country', 'ds': 'Year', 'OptimizedTheta': 'Prediction'})
    predict.index = predict.index + 1
    return predict

show_graph()
print(predict())
