from flask import Flask, render_template
import plotly.graph_objects as go
import requests
import datetime
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from plotly.offline import plot
import plotly.express as px
import math
import numpy as np



#INput 

start = datetime(2015, 12, 1)
p1 = {"AAPL" : 0.2 , "IBM" : 0.2 , "NVDA" : 0.2 , "AMZN" : 0.2 , "GOOG":0.2 }

#Functions 

def get_stock_value_from_ticker(ticker , start):

  data = yf.download(ticker, start=start)
  columns_to_drop = ['Open', 'High' ,"Low","Close","Volume" ]
  df = data.drop(columns=columns_to_drop)

  return df

def get_prices_portefeuille(portefeuille , start):
  return { ticker : get_stock_value_from_ticker(ticker , start) for ticker in portefeuille.keys() }

def get_portefeuille_value(portefeuille , start , start_capital):
  actions = get_prices_portefeuille(portefeuille , start)

  weighted = pd.DataFrame({})

  for action in actions.keys():
    weighted[action] = actions[action] * portefeuille[action] * start_capital / actions[action]["Adj Close"][0]

  weighted_filtered = weighted.dropna(axis=0, how='any')

  weighted_filtered["somme"] =  weighted_filtered.sum(axis=1)

  return weighted_filtered


def get_portefeuille_avec_mensualite(portefeuille , start , start_capital , monthly_payment) :
  df = get_portefeuille_value(portefeuille , start , start_capital)

  df["somme mensu"] = df["somme"]
  df["somme investi"] = start_capital

  date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='MS') + pd.DateOffset(days=(start.day ))
  for date in date_range[1:]:
      date_str = date.strftime('%Y-%m-%d')
      df.loc[ df.index > date_str , "somme mensu" ] *= ( 1 + (monthly_payment / df["somme mensu"][date_str :].iloc[0] )) 
      df.loc[ df.index > date_str , "somme investi" ] += monthly_payment
  return df


def calculate_future_value(initial_sum, monthly_payment, annual_interest_rate, years):
    # Convert annual interest rate to decimal and calculate monthly interest rate
    monthly_interest_rate = (annual_interest_rate / 100) / 12

    # Calculate the number of compounding periods
    compounding_periods = 12

    # Calculate the total number of payments
    total_payments = compounding_periods * years

    # Generate dates for each month
    date_today = datetime.now()
    date_list = [(date_today + timedelta(days=30.44 * i)).date() for i in range(1, total_payments + 1)]

    # Calculate the future value using the formula
    future_value = [initial_sum * (1 + monthly_interest_rate)**(compounding_periods * i) +
                    monthly_payment * (((1 + monthly_interest_rate) ** (compounding_periods * i)) - 1) / monthly_interest_rate
                    for i in range(1, total_payments + 1)]

    # Create a DataFrame
    df = pd.DataFrame(future_value, index=date_list, columns=["Future Value"])

    return df

def calculate_cagr(df):
    initial_date = (df.index[0].date())
    final_date = (df.index[-1].date())


    initial_value = (df["somme"][0])
    final_value = (df["somme"][-1])



    # Calculate the difference in months
    months_difference = (final_date.year - initial_date.year) * 12 + final_date.month - initial_date.month
    years = months_difference / 12;
    # Calculer le CAGR
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr

def show_portefeuille_value(df):
  fig = go.Figure([go.Scatter(x=df.index , y=df['somme'])])
  plot_html = plot(fig, output_type='div')
  return plot_html


def show_portefeuille_value_mensu(df):
  fig = go.Figure([go.Scatter(x=df.index , y=df["somme mensu"] , name="avec mensualité") , go.Scatter(x=df.index , y=df["somme investi"],name="somme investi") , go.Scatter(x=df.index , y=df['somme'] , name="sans mensualité")])
  plot_html = plot(fig, output_type='div')
  return plot_html




def show_somme_investi(df):
  fig = go.Figure([go.Scatter(x=df.index , y=df["somme investi"])])
  plot_html = plot(fig, output_type='div')
  return plot_html



def compute_metrics(df , risk_free_rate):

  initial_value = (df["somme"][0])
  final_value = (df["somme"][-1])
  years_length = (df.index[len(df)-1] - df.index[0] ).days / 365
  trading_days = len(df)
  day_rate = risk_free_rate*years_length/trading_days
  # Calculate daily returns
  df['Daily returns'] = df['somme'].pct_change()

  # Calculate average bi weekly return and standard deviation of bi weekly return
  average_return = df['Daily returns'].mean()
  
  std_dev_return = df['Daily returns'].std()
  annual_volatility = std_dev_return*math.sqrt(252)


  cagr = (final_value / initial_value) ** (1 / years_length) - 1

  # Calculate Sharpe ratio
  sharpe_ratio = (average_return - day_rate )/ std_dev_return
  annualized_sharpe_ratio =   sharpe_ratio*math.sqrt(365)
  return {'annual_volatility' : annual_volatility , 'volatility' : std_dev_return , "sharpe_ratio" : annualized_sharpe_ratio , "cagr" : cagr ,"average_return" : average_return * 252 }




app = Flask(__name__)

@app.route('/')
def index():

    # Get the data from the API
    df = get_portefeuille_avec_mensualite(p1 , start , 1000,100)

    # Convert the figure to HTML
    plot_html = show_portefeuille_value_mensu(df)

    return render_template('index.html', plot=plot_html)


@app.route('/metrics')
def metrics():
  df = get_portefeuille_avec_mensualite(p1 , start , 1000,100)
  res = compute_metrics(df , 0.01)
  return res

if __name__ == '__main__':
    app.run(debug=True)
