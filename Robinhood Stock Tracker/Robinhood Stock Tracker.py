"""Robinhood Stock Tracker"""

"""Goal:

Create an app that, via an end of day report:
1. Keeps track of my Robinhood stocks or assets
2. Informs me when a stock in my portfolio is underperforming
3. Informs me when a stock in my portfolio is doing well
4. Advise me on next steps for a stock, whether to buy or sell. 
5. Provides stock insights for my portfolio

More advanced stage:
6. Predicts the 1 week perfomance of my stocks
"""
#----------------------------------------------------------------------------------------------
"""Import Packages"""
#----------------------------------------------------------------------------------------------
import pandas as pd 
import numpy as np
import pandas_datareader as web
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date 
import pendulum
import random

import bs4
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen 
import requests

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table

import math 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
#----------------------------------------------------------------------------------------------
"""Defining Portfolio, Creating Webscraping URLs"""
#----------------------------------------------------------------------------------------------
#define stock portfolio
portfolio_data = {
    "Stock Name": ["Sirius XM Holdings Inc", "Stratasys Ltd", "Pacific Biosciences of California, Inc", "Mondelez International, Inc", "The Unilever Group", "The Unilever Group."],
    "Symbol": ["SIRI", "SSYS", "PACB", "MDLZ", "UL", "UN"],
    "Purchase Price": [6.23, 21.39, 7.82, 44.98, 52.25, 53.80]
}
portfolio = pd.DataFrame(portfolio_data)
portfolio = portfolio.set_index("Symbol")

#define URLs to (1) webscrape for stock information (2) download historical data
#stock summary url
stock_url_component_1 = "https://finance.yahoo.com/quote/"
stock_url_component_2 = "/?p="

#stock historical data
historical_data_url_1 = "https://query1.finance.yahoo.com/v7/finance/download/"
historical_data_url_2 = "?period1="  #yahoo finance uses timestamps for its searches
historical_data_url_3 = "&period2="
historical_data_url_4 = "&interval=1d&events=history"

#define dates for historical data - interested in 1 year data from yesterday
yesterday = datetime.today() - timedelta(days=1)
yesterday = datetime(yesterday.year, yesterday.month, yesterday.day, 20, 0)
yesterday_timestamp = round(datetime.timestamp(yesterday))

last_year_yesterday = datetime.today() - timedelta(days=364)
last_year_yesterday = datetime(last_year_yesterday.year, last_year_yesterday.month, last_year_yesterday.day, 20, 0)
last_year_yesterday_timestamp = round(datetime.timestamp(last_year_yesterday))

stock_urls = []
stock_historical_data_url = []
for (i, j) in zip(portfolio.index.tolist(), portfolio.index.tolist()):
    #get URL to stock main/ summary page - for future web scraping
    url_1 = stock_url_component_1+i+stock_url_component_2+i
    stock_urls.append(url_1)

    #get url to download historical data, for plots and other insights
    url_2 = historical_data_url_1+i+historical_data_url_2+str(last_year_yesterday_timestamp)+historical_data_url_3+str(yesterday_timestamp)+historical_data_url_4
    stock_historical_data_url.append(url_2)

portfolio["Stock Summary URLs"] = stock_urls
portfolio["Stock Historical Data URLs"] = stock_historical_data_url
#----------------------------------------------------------------------------------------------
"""Data Collection, Cleaning and Processing"""
#----------------------------------------------------------------------------------------------
#stock summary info: current price, previous closing price, opening pricem today's volume, average volume, market cap, beta, PE ratio and 1-year target
current_price = []
previous_closing_price = []
opening_price = []
todays_volume = []
average_volume = []
market_cap = []
beta = []
pe_ratio = []
year_target = []
#historical data
historical_data = []
for (i, j) in zip(portfolio["Stock Summary URLs"].unique().tolist(), portfolio["Stock Historical Data URLs"].unique().tolist()):
    #get closing/current stock price via BeautifulSoup
    webpage = urlopen(i)
    page_html = webpage.read()
    webpage.close()
    #parse HTML
    webpage_soup = BeautifulSoup(page_html, "html.parser")
    price_1 = webpage_soup.findAll("div", {"class": "My(6px) Pos(r) smartphone_Mt(6px)"})[0].span.text.replace(",", "")
    current_price.append(price_1)
    price_2 = webpage_soup.findAll("td", {"data-test": "PREV_CLOSE-value"})[0].span.text.replace(",", "")
    previous_closing_price.append(price_2)
    price_3 = webpage_soup.findAll("td", {"data-test": "OPEN-value"})[0].span.text.replace(",", "")
    opening_price.append(price_3)
    volume_1 = webpage_soup.findAll("td", {"data-test": "TD_VOLUME-value"})[0].text.replace(",", "")
    todays_volume.append(volume_1)
    volume_2 = webpage_soup.findAll("td", {"data-test": "AVERAGE_VOLUME_3MONTH-value"})[0].text.replace(",", "")
    average_volume.append(volume_2)
    mk_cap = webpage_soup.findAll("td", {"data-test": "MARKET_CAP-value"})[0].text
    market_cap.append(mk_cap)
    bt = webpage_soup.findAll("td", {"data-test": "BETA_5Y-value"})[0].text.replace(",", "")
    beta.append(bt)
    pe_r = webpage_soup.findAll("td", {"data-test": "PE_RATIO-value"})[0].text.replace(",", "")
    pe_ratio.append(pe_r)	
    target = webpage_soup.findAll("td", {"data-test": "ONE_YEAR_TARGET_PRICE-value"})[0].text.replace(",", "")
    year_target.append(target)

    #get historical data of each stock
    data = pd.read_csv(j)
    historical_data.append(data)

#add to portfolio dataframe
portfolio["Current Price"] = current_price
portfolio["Previous Closing Price"] = previous_closing_price
portfolio["Opening Price"] = opening_price
portfolio["Volume (today)"] = todays_volume
portfolio["Average Volume"] = average_volume
portfolio["Market Cap"] = market_cap
portfolio["Beta (5Y Monthly)"] = beta
portfolio["PE Ratio (TTM)"] = pe_ratio
portfolio["1-Year Target"] = year_target
portfolio["Historical Data"] = historical_data

#get previous day Adjusted Closing prices
previous_day_closing_price = []
for i in portfolio.index.tolist():
    price = portfolio.loc[i, "Historical Data"].tail(1).iloc[0]["Adj Close"]  
    previous_day_closing_price.append(round(price, 2))

portfolio["Previous Day Adjusted Closing Price"] = previous_day_closing_price

#----------------------------------------------------------------------------------------------
"""Data Analysis

2. Inform low-performing stocks
3. Inform high-performing stocks
4. Advise on next steps
5. Stock Insights

Advanced:
6. Predict 1 week stock perfomance
"""
#----------------------------------------------------------------------------------------------
"""Stock Performance:
Assessed by comparison to purchace price and closing price from prior day

Stock advising from:
https://www.investopedia.com/investing/selling-a-losing-stock/
https://www.investors.com/ibd-university/how-to-sell/limit-losses/
"""
portfolio["Current Price"] = portfolio["Current Price"].astype(float)
portfolio["Purchase Price"] = portfolio["Purchase Price"].astype(float)
portfolio["Previous Day Adjusted Closing Price"] = portfolio["Previous Day Adjusted Closing Price"].astype(float)
portfolio["% Change (Purchace Price)"] = round((portfolio["Current Price"] - portfolio["Purchase Price"])*100/(portfolio["Current Price"]), 2)
portfolio["% Change (Previous Closing Price)"] = round((portfolio["Current Price"] - portfolio["Previous Day Adjusted Closing Price"])*100/(portfolio["Current Price"]), 2)

#generally, its good to keep losses under 8 %. I am new and broke, so I'll keep my losses to under 6 % (from purchasing price)
#underperorming stocks
bad_stocks = portfolio[portfolio["% Change (Purchace Price)"] < -6.00].index.tolist()
print(f"The following stocks are underperforming: {bad_stocks}. Please look closely into them and consider a potential sale.")
#keep track of well-performing stocks
#an increase of 10 % is good, and we can decide further of we want to sell or purchase more of the stock
#keep in mind: better to get a stock at a lower price as it rises in value, compared to mid-rise or at its peak
good_stocks = portfolio[portfolio["% Change (Purchace Price)"] > 10.00].index.tolist()
print(f"The following stocks are performing well: {good_stocks}. Consider selling for profit, or purchasing more before price becomes too high")

#stock summary table:
summary_data_initialize = {
"Category": ["Symbol", "Purchase Price ($ USD)", "Opening Price ($ USD)", "Previous Closing Price ($ USD)", "Current Volume", "Average Volume", "Market Cap ($ USD)", "Beta ($ USD)", "PE Ratio ($ USD)", "1-Year Target ($ USD)"],
"Summary": ["", "", "", "", "", "", "", "", "", ""]
}
summary_tab = pd.DataFrame(summary_data_initialize)

#predictive model - Using a neural network to predict stock price using a years data 
components_train = []
components_valid = []
for j in portfolio.index.tolist():
    overall_data = web.DataReader(j, data_source = "yahoo", start = "2010-01-01", end = yesterday.strftime("%Y-%m-%d"))
    dataset = overall_data.filter(["Adj Close"])
    dataset = dataset.values
    #determine data length
    train_data_len = math.ceil(len(dataset) * 0.8)
    #scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #create training dataset 
    train_data = scaled_data[0: train_data_len, :]
    #split into x_train and y_train
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    #Test data set
    test_data = scaled_data[train_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[train_data_len : , : ]

    for k in range(60,len(test_data)):
        x_test.append(test_data[k-60:k,0])

    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)#Undo scaling
 
    train = overall_data[:train_data_len]
    components_train.append(train)
    valid = overall_data[train_data_len:]
    valid['Predictions'] = predictions
    components_valid.append(valid)

portfolio["ML Train Data"] = components_train
portfolio["ML Prediction"] = components_valid

#----------------------------------------------------------------------------------------------
"""Initialize Dashboard Application"""
#----------------------------------------------------------------------------------------------
app = dash.Dash(__name__)
#----------------------------------------------------------------------------------------------
"""Application Layout"""
#----------------------------------------------------------------------------------------------
app.layout = html.Div([

    #header
    html.Div(
        html.P("Robinhood Stock Tracker", 
        style = {"font-family": "Courier New", "font-size": "45px", "font-weight": "bold", "text-align": "center", "color": "white"}), 
        style = {"background-color": "#0373fc", "display": "inline-block", "width": "100%"}
    ),

    #first row
    html.Div([
        #section 1: selection tool
        html.Div([
            html.Br(),
            html.Br(),
            html.Br(),

            html.Div(html.P("Select Stock Option", style = {"font-family": "Courier New", "font-size": "20px", "text-align": "center"})),

            html.Div(dcc.Dropdown(id = "stock_dropdown",
                                    options = [
                                        {"label": i, "value": i} for i in portfolio.index.tolist()
                                    ],
                                    multi = False,
                                    value = "SIRI", style = {"font-family": "Courier New", "font-size": "15px", "text-align": "center"}))
        ], style = {"background-color": "#074db0"}, className = "two columns"),

        #section 2: plot area
        html.Div(dcc.Graph(id = "stock_graph", figure = [], style = {"height": "350px"}), className = "six columns"),

        #section 3: summary 
        html.Div([
            html.P(f"Stock Summary", style = {"font-family": "Courier New", "font-size": "20px", "text-align": "center", "text-decoration": "underline"}),

            html.Div(dash_table.DataTable(
                id = "summary_table",
                columns = [{"name": i, "id": i} for i in summary_tab.columns],
                style_table = {"height": "300px", "overflowY": "auto"},
                page_action = "none",
                style_cell = {'textAlign': 'left', "color": "black"},
                style_header = {"font-weight": "bold", 'border': '2px solid grey', "background-color": "#074db0", "text-align": "center"},
                style_data = {'border': '2px solid grey', "background-color": "#074db0"}
            ), style = {"text-align": "center"})
        ], style = {"background-color": "#074db0"}, className = "four columns")

], className = "row", style = {"background-color": "#074db0", "height": "350px"}),

    #second row
    html.Div([
        #section 1: selection tool
        html.Div([
            html.Br(),
            html.Br(),
            html.Br(),

            html.Div(html.P("Select Perfomance Category", style = {"font-family": "Courier New", "font-size": "20px", "text-align": "center"})),

            html.Div(dcc.Dropdown(id = "performance_dropdown",
                                    options = [
                                        {"label": "Good Stocks", "value": "Good Stocks"},
                                        {"label": "Bad Stocks", "value": "Bad Stocks"}
                                    ],
                                    multi = False,
                                    value = "Good Stocks", style = {"font-family": "Courier New", "font-size": "15px", "text-align": "center"}))
        ], style = {"background-color": "#074f8f"}, className = "two columns"),

        #section 2 : plot area
        html.Div(dcc.Graph(id = "performance_graph", figure = [], style = {"height": "350px"}), className = "four columns"),

        #section 3: selection tool - predictive model
        html.Div([
            html.Br(),
            html.Br(),
            html.Br(),

            html.Div(html.P("Select Stock To Predict", style = {"font-family": "Courier New", "font-size": "20px", "text-align": "center"})),

            html.Div(dcc.Dropdown(id = "prediction_dropdown",
                                    options = [{"label": i, "value": i} for i in portfolio.index.tolist()], 
                                    multi = False,
                                    value = "SIRI", style = {"font-family": "Courier New", "font-size": "15px", "text-align": "center"}))
        ], style = {"background-color": "#074f8f"}, className = "two columns"),

        #section 4: plot area - stock prediction
        html.Div(dcc.Graph(id = "prediction_graph", figure = [], style = {"height": "350px"}), className = "four columns")
], className = "row", style = {"background-color": "#074f8f", "height": "350px"}
    )], className = "twelve columns")

#----------------------------------------------------------------------------------------------
"""Define Callbacks"""
#----------------------------------------------------------------------------------------------
#graph area 1 - historical data
@app.callback(
    Output(component_id = "stock_graph", component_property = "figure"),
    [Input(component_id = "stock_dropdown", component_property = "value")]
)
def update_stock_graph(stock_selected):
    #set x and y data
    x = portfolio.loc[stock_selected, "Historical Data"]["Date"].tolist()
    x_rev = x[::-1]
    y = portfolio.loc[stock_selected, "Historical Data"]["Close"].tolist()
    y_upper = portfolio.loc[stock_selected, "Historical Data"]["High"].tolist()
    y_lower = portfolio.loc[stock_selected, "Historical Data"]["Low"].tolist()
    y_lower = y_lower[::-1]

    #initialize figure
    fig_1 = go.Figure()

    #add data as scatter plot
    fig_1.add_trace(go.Scatter(
        x = x_rev+x,
        y = y_lower+y_upper,
        fill='toself',
        fillcolor='rgba(182, 240, 228, 0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name = stock_selected,
    ))
    fig_1.add_trace(go.Scatter(
        x=x, y=y,
        line_color='rgb(15, 189, 153)',
        name = stock_selected,
    ))
    #modify to line plots
    fig_1.update_traces(mode='lines')
    fig_1.update_layout(
        title = f"Historical Data: {portfolio.loc[stock_selected, 'Stock Name']} ({stock_selected})",
        xaxis_title = "Date",
        yaxis_title = "Adj Close ($ USD)",
        template = "plotly_dark",
    )
    return fig_1

#graph area 2 - good and bad stocks
@app.callback(
    Output(component_id = "performance_graph", component_property = "figure"),
    [Input(component_id = "performance_dropdown", component_property = "value")]
)
def update_performance_graph(option_selected):
    if option_selected == "Good Stocks":
        fig_2 = go.Figure()
        for i in good_stocks:
            fig_2.add_trace(go.Scatter(
            x=portfolio.loc[i, "Historical Data"]["Date"], 
            y=portfolio.loc[i, "Historical Data"]["Adj Close"],
            line_color = random.choice(['rgb(0, 100, 80)', 'rgb(184, 42, 42)', 'rgb(19, 176, 40)', 'rgb(230, 141, 25)', 'rgb(20, 54, 204)']),
            name= i,
            ))
            fig_2.add_shape(
                # Line Horizontal
                dict(
                    type = "line",
                    x0 = portfolio.loc[i, "Historical Data"]["Date"].min(),
                    y0 = portfolio.loc[i, "Purchase Price"],
                    x1 = portfolio.loc[i, "Historical Data"]["Date"].max(),
                    y1 = portfolio.loc[i, "Purchase Price"],
                    line = dict(
                        color = random.choice(["red", "blue", "green", "purple", "orange"]),
                        width = 2,
                        dash = "dot")
                ))
        fig_2.update_layout(
            title = {
                "text": f"High Performing Stocks Relative to Purchase Price <br> (Δ > 10 %): {', '.join(good_stocks)}",
                "x": 0.5,
                "y": 0.85,
                "font": dict(size = 15),
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template = "plotly_dark",
            xaxis_title = "Date",
            yaxis_title = "Adj Close ($ USD)")
        fig_2.update_shapes(dict(xref = 'x', yref = 'y'))        
    else:
        fig_2 = go.Figure()
        for i in bad_stocks:
            fig_2.add_trace(go.Scatter(
            x=portfolio.loc[i, "Historical Data"]["Date"], 
            y=portfolio.loc[i, "Historical Data"]["Adj Close"],
            line_color = random.choice(['rgb(0, 100, 80)', 'rgb(184, 42, 42)', 'rgb(19, 176, 40)', 'rgb(230, 141, 25)', 'rgb(20, 54, 204)']),
            name= i,
            ))
            fig_2.add_shape(
                # Line Horizontal
                dict(
                    type = "line",
                    x0 = portfolio.loc[i, "Historical Data"]["Date"].min(),
                    y0 = portfolio.loc[i, "Purchase Price"],
                    x1 = portfolio.loc[i, "Historical Data"]["Date"].max(),
                    y1 = portfolio.loc[i, "Purchase Price"],
                    line = dict(
                        color = random.choice(["red", "blue", "green", "purple", "orange"]),
                        width = 2,
                        dash = "dot")
                ))
        fig_2.update_layout(
            title = {
                "text": f"Low Performing Stocks Relative to Purchase Price <br> (Δ < 6 %): {', '.join(bad_stocks)}",
                "x": 0.5,
                "font": dict(size = 15),
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template = "plotly_dark",
            xaxis_title = "Date",
            yaxis_title = "Adj Close ($ USD)")
        fig_2.update_shapes(dict(xref = 'x', yref = 'y'))
    
    return fig_2

#stock summary
@app.callback(
    Output(component_id = "summary_table", component_property = "data"),
    [Input(component_id = "stock_dropdown", component_property = "value")]
)
def update_summary_table(stock_selected_summary):
    
    summary_data = {
    "Category": ["Symbol", "Purchase Price ($ USD)", "Opening Price ($ USD)", "Previous Closing Price ($ USD)", "Current Volume", "Average Volume", "Market Cap ($ USD)", "Beta ($ USD)", "PE Ratio ($ USD)", "1-Year Target ($ USD)"],
    "Summary": [stock_selected_summary, portfolio.loc[stock_selected_summary, "Purchase Price"], portfolio.loc[stock_selected_summary, "Opening Price"], portfolio.loc[stock_selected_summary, "Previous Closing Price"], portfolio.loc[stock_selected_summary, "Volume (today)"],
    portfolio.loc[stock_selected_summary, "Average Volume"], portfolio.loc[stock_selected_summary, "Market Cap"], portfolio.loc[stock_selected_summary, "Beta (5Y Monthly)"], portfolio.loc[stock_selected_summary, "PE Ratio (TTM)"], portfolio.loc[stock_selected_summary, "1-Year Target"]]
    }

    summary = pd.DataFrame(summary_data)

    return summary.to_dict("records")

#predictive model
@app.callback(
    Output(component_id = "prediction_graph", component_property = "figure"),
    [Input(component_id = "prediction_dropdown", component_property = "value")]
)

def update_predictive_graph(prediction_selected):
    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(
        x=portfolio.loc[prediction_selected, "ML Train Data"].index, 
        y=portfolio.loc[prediction_selected, "ML Train Data"]["Adj Close"],
        line_color = 'rgb(30, 182, 232)',
        name= f"Training Data ({prediction_selected})"
    ))
    fig_3.add_trace(go.Scatter(
        x=portfolio.loc[prediction_selected, "ML Prediction"].index, 
        y=portfolio.loc[prediction_selected, "ML Prediction"]['Predictions'],
        line_color = 'rgb(232, 155, 30)',
        name= "Predicted"
    ))
    fig_3.add_trace(go.Scatter(
        x=portfolio.loc[prediction_selected, "ML Prediction"].index, 
        y=portfolio.loc[prediction_selected, "ML Prediction"]['Adj Close'],
        line_color = 'rgb(20, 204, 94)',
        name= "Actual Value"
    ))
    fig_3.update_layout(
        title = {
            "text": f"Predictive Model using LSTM (10-Year Data): {prediction_selected}",
            "x": 0.5,
            "font": dict(size = 15),
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template = "plotly_dark",
        xaxis_title = "Date",
        yaxis_title = "Adj Close ($ USD)")
    fig_3.update_shapes(dict(xref = 'x', yref = 'y'))

    return fig_3
#----------------------------------------------------------------------------------------------
"""Run Application"""
#----------------------------------------------------------------------------------------------
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1')








