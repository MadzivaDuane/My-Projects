"""Yahoo Finance Stock Watchlist"""

"""
Goal:

Create an Dash Visualization App responsible for:
1. Aggregating information on a stock's performance
2. Providing layered technical indicators to make a buying or selling decision 
3. Comparing historical performance and technical indicators of 2 stocks
"""

#----------------------------------------------------------------------------------------------
"""Import Packages"""
#----------------------------------------------------------------------------------------------
import pandas as pd, pandas_datareader as web, datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date 

import dash, dash_core_components as dcc, dash_html_components as html
from dash.dependencies import Input, Output

import yfinance as yf
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
        html.P("Yahoo Finance Stock Watchlist", 
        style = {"font-family": "Courier New", "font-size": "30px", "font-weight": "bold", "text-align": "center", "color": "white"}), 
        style = {"background-color": "#0f58f7", "display": "inline-block", "width": "100%"}),
    #search bar 1: ticker, start date, end date, MA and RSI
    html.Div([
        dcc.Input(
            id = "search_box_1", placeholder='Enter a Stock Ticker/ Symbol...', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "30%"}
            ),
        dcc.Input(
            id = "start_date_1", placeholder='Start Date (MM/DD/YYY)', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "20%"}
            ),
        dcc.Input(
            id = "end_date_1", placeholder= 'End Date (MM/DD/YYY)', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "20%"})
        ], className = "row"),
    #graphs of first stock selection
    html.Div([
        #stock summary graph
        html.Div(dcc.Graph(id = "stock_summary_1", figure = [], style = {"height": "350px"}), className = "five columns"),
        #delta MAs graph
        html.Div(dcc.Graph(id = "delta_ma_1", figure = [], style = {"height": "350px"}), className = "three columns"),
        #RSI graph
        html.Div(dcc.Graph(id = "rsi_1", figure = [], style = {"height": "350px"}), className = "four columns")
    ], className = "row"),
    #search bar 2: ticker, start date and end date
    html.Div([
        dcc.Input(
            id = "search_box_2", placeholder='Enter a Stock Ticker/ Symbol...', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "30%"}
            ),
        dcc.Input(
            id = "start_date_2", placeholder='Start Date (MM/DD/YYYY)', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "20%"}
            ),
        dcc.Input(
            id = "end_date_2", placeholder= 'End Date (MM/DD/YYYY)', type='text', value='', style = {"font-family": "Courier", "font-size": "15px", "display": "inline-block", "width": "20%"}
            )
            ] , className = "row"),
    #graphs of second stock selection
    html.Div([
        #stock summary graph
        html.Div(dcc.Graph(id = "stock_summary_2", figure = [], style = {"height": "350px"}), className = "five columns"),
        #delta MAs graph
        html.Div(dcc.Graph(id = "delta_ma_2", figure = [], style = {"height": "350px"}), className = "three columns"),
        #RSI graph
        html.Div(dcc.Graph(id = "rsi_2", figure = [], style = {"height": "350px"}), className = "four columns")
    ], className = "row")], className = "twelve columns") 
#----------------------------------------------------------------------------------------------
"""Define Callbacks"""
#----------------------------------------------------------------------------------------------
#stock summary 1
@app.callback(
    Output(component_id = "stock_summary_1", component_property = "figure"),
    [Input(component_id = "search_box_1", component_property = "value"),
    Input(component_id = "start_date_1", component_property = "value"),
    Input(component_id = "end_date_1", component_property = "value"),]
)
def update_stock_graph_1(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate the moving averages (MAs) - weekly trader so I'll focus on 5, 10 and 20 day MAs
    stock_data["5-day MA"] = round(stock_data['Adj Close'].rolling(window = 5).mean(), 6)
    stock_data["10-day MA"] = round(stock_data['Adj Close'].rolling(window = 10).mean(), 6)
    stock_data["20-day MA"] = round(stock_data['Adj Close'].rolling(window = 20).mean(), 6)
    #calculate 14-day RSI
    rsi_window_length_1 = 14  #default is 14 days
    close = stock_data["Adj Close"]
    #Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the SMA - simple moving average
    #using 14 day interval
    roll_up_14 = up.rolling(rsi_window_length_1).mean()
    roll_down_14 = down.abs().rolling(rsi_window_length_1).mean()
    # Calculate the RSI based on SMA
    RS_14 = roll_up_14 / roll_down_14
    stock_data["14-day RSI"] = 100.0 - (100.0 / (1.0 + RS_14))
    #plot figure
    fig_1 = make_subplots(rows = 5, cols = 1, shared_xaxes=True, vertical_spacing=0.00625, specs = [[{"rowspan": 3}], [None], [None], [{}], [{}]])
    #add traces - Candlestick, 20-day MA, 40-day MA, RSI, and Volume
    fig_1.add_trace(
        go.Candlestick(
            x = stock_data.index, open = stock_data["Open"], high = stock_data["High"], low = stock_data["Low"], close = stock_data["Close"],
            name = "Price", increasing_line_color = "green", increasing = dict(line = dict(color = "black"))), row = 1, col = 1)
    fig_1.update_layout(xaxis_rangeslider_visible = False)  #disable slider
    #Add 5 and 10-day moving averages
    fig_1.add_trace(
    go.Scatter(
        x = stock_data.index, y = stock_data["5-day MA"], mode = "lines", name = "5-day MA", line = dict(color = "blue")))
    fig_1.add_trace(
    go.Scatter(
        x = stock_data.index, y = stock_data["10-day MA"], mode = "lines", name = "10-day MA", line = dict(color = "orange")))
    #add RSI
    fig_1.add_scatter(
        x = stock_data.index, y = stock_data["14-day RSI"], mode = "lines", name = "14-day RSI", line = dict(color = "purple"), row = 4, col = 1)
    fig_1.update_layout(
        shapes = [
            dict(type="line", xref="x2", yref="y2", line = dict(color = "green", dash = "dot"),
                x0=min(stock_data.index), y0=30, x1=max(stock_data.index), y1=30),
            dict(type="line", xref="x2", yref='y2', line = dict(color = "red", dash = "dot"),
                x0=min(stock_data.index), y0=70, x1=max(stock_data.index), y1=70)])
    #add volume
    fig_1.add_bar(
        x = stock_data.index, y = stock_data["Volume"], name = "Volume", row = 5, col = 1)
    #add y-axis titles for top 2 components of combined plot
    fig_1.update_yaxes(title_text="($ USD)", row=1, col=1)
    fig_1.update_yaxes(title_text="(%)", row=4, col=1)

    fig_1.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke",
    margin=dict(l=5, t=40, b=20), title_text= f"Stock: {stock_selected.upper()}")
    fig_1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    
    return fig_1

#difference in 5 and 10-day MAs plot
@app.callback(
    Output(component_id = "delta_ma_1", component_property = "figure"),
    [Input(component_id = "search_box_1", component_property = "value"),
    Input(component_id = "start_date_1", component_property = "value"),
    Input(component_id = "end_date_1", component_property = "value"),]
)
def update_delta_MA_graph_1(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate the moving averages (MAs) - weekly trader so I'll focus on 5, 10 and 20 day MAs
    stock_data["5-day MA"] = round(stock_data['Adj Close'].rolling(window = 5).mean(), 6)
    stock_data["10-day MA"] = round(stock_data['Adj Close'].rolling(window = 10).mean(), 6)
    stock_data["20-day MA"] = round(stock_data['Adj Close'].rolling(window = 20).mean(), 6)
    #weekly delta MA insight
    weekly_data = stock_data.tail(14)
    weekly_data["5 - 15 day MA (%)"] = (weekly_data["5-day MA"] - weekly_data["10-day MA"])*100/weekly_data["5-day MA"]
    #Change in Moving Average Crossover (Difference between 2 MAs)
    fig_2 = go.Figure()
    fig_2.add_trace(
        go.Scatter(
            x = weekly_data.index, y = weekly_data["5 - 15 day MA (%)"], mode = "lines", name = "Δ MA (%)", line = dict(color = "royalblue")))
    fig_2.update_layout(
        shapes = [
            dict(type = "line", line = dict(color = "green", dash = "dot"), 
            x0 = min(weekly_data.index), y0 = 0.1, x1 = max(weekly_data.index), y1 = 0.1),
            dict(type = "line", line = dict(color = "red", dash = "dot"), 
            x0 = min(weekly_data.index), y0 = 0.5, x1 = max(weekly_data.index), y1 = 0.5),
            dict(type = "line", line = dict(color = "lightgrey", width = 0.05), 
            x0 = min(weekly_data.index), y0 = 0, x1 = max(weekly_data.index), y1 = 0)])
    # Create scatter trace of text labels
    fig_2.add_trace(go.Scatter(
        x=[min(weekly_data.index)+((max(weekly_data.index)-min(weekly_data.index))/2), min(weekly_data.index)+((max(weekly_data.index)-min(weekly_data.index))/2)],
        y=[0.05, 0.45], text=["Min Buy Signal", "Max Sell Signal"], mode="text", showlegend=False))
    fig_2.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke", title_x=0.5,
    margin=dict(l=5, t=40, b=20), title_text= f"2-Week Δ 5 & 10-day MAs (%), {stock_selected.upper()}")
    fig_2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    
    return fig_2

#5 and 14-day RSIs plot
@app.callback(
    Output(component_id = "rsi_1", component_property = "figure"),
    [Input(component_id = "search_box_1", component_property = "value"),
    Input(component_id = "start_date_1", component_property = "value"),
    Input(component_id = "end_date_1", component_property = "value"),]
)
def update_delta_RSI_graph_1(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate 14-day RSI
    rsi_window_length_1 = 14  #default is 14 days
    rsi_window_length_2 = 5    #for a much more robust response to price changes
    close = stock_data["Adj Close"]
    #Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the SMA - simple moving average
    #using 14 day interval
    roll_up_14 = up.rolling(rsi_window_length_1).mean()
    roll_down_14 = down.abs().rolling(rsi_window_length_1).mean()
    #using 5 day interval
    roll_up_5 = up.rolling(rsi_window_length_2).mean()
    roll_down_5 = down.abs().rolling(rsi_window_length_2).mean()
    # Calculate the RSI based on SMA
    RS_14 = roll_up_14 / roll_down_14
    RS_5 = roll_up_5 / roll_down_5
    stock_data["14-day RSI"] = 100.0 - (100.0 / (1.0 + RS_14))
    stock_data["5-day RSI"] = 100.0 - (100.0 / (1.0 + RS_5))
    monthly_rsi = stock_data[["14-day RSI", "5-day RSI"]].tail(31)
    fig_3 = go.Figure()
    #add the 2 RSI values (5 and 14-day)
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"], line = {"color": "purple"}, name = "14-day")
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"], line = {"color": "blue"}, name = "5-day")
    #color change for intervals
    #14-day RSI
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"].where(monthly_rsi["14-day RSI"] <= 30), line = {"color": "green"}, showlegend=False)
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"].where(monthly_rsi["14-day RSI"] >= 70), line = {"color": "red"}, showlegend=False)
    #5-day interval
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"].where(monthly_rsi["5-day RSI"] <= 30), line = {"color": "green"}, showlegend=False)
    fig_3.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"].where(monthly_rsi["5-day RSI"] >= 70), line = {"color": "red"}, showlegend=False)

    #thresholds: 70 % overbought, 30 % oversold
    fig_3.update_layout(
        shapes = [
            dict(type = "line", line = dict(color = "green", dash = "dot"), 
            x0 = min(monthly_rsi.index), y0 = 30, x1 = max(monthly_rsi.index), y1 = 30),
            dict(type = "line", line = dict(color = "red", dash = "dot"), 
            x0 = min(monthly_rsi.index), y0 = 70, x1 = max(monthly_rsi.index), y1 = 70)])
    # Create scatter trace of text labels
    fig_3.add_trace(go.Scatter(
        x=[min(monthly_rsi.index)+timedelta(days=5), min(monthly_rsi.index)+timedelta(days=5)],
        y=[25, 65], text=["RSI = 30 %", "RSI = 70 %"], mode="text", showlegend=False))

    fig_3.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke", title_x=0.5,
    margin=dict(l=5, t=40, b=20), title_text= f"Monthly 5 & 14-day RSI (%), {stock_selected.upper()}")
    fig_3.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_3.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")

    return fig_3

#stock summary plot 2
@app.callback(
    Output(component_id = "stock_summary_2", component_property = "figure"),
    [Input(component_id = "search_box_2", component_property = "value"),
    Input(component_id = "start_date_2", component_property = "value"),
    Input(component_id = "end_date_2", component_property = "value"),]
)
def update_stock_graph_2(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate the moving averages (MAs) - weekly trader so I'll focus on 5, 10 and 20 day MAs
    stock_data["5-day MA"] = round(stock_data['Adj Close'].rolling(window = 5).mean(), 6)
    stock_data["10-day MA"] = round(stock_data['Adj Close'].rolling(window = 10).mean(), 6)
    stock_data["20-day MA"] = round(stock_data['Adj Close'].rolling(window = 20).mean(), 6)
    #calculate 14-day RSI
    rsi_window_length_1 = 14  #default is 14 days
    close = stock_data["Adj Close"]
    #Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the SMA - simple moving average
    #using 14 day interval
    roll_up_14 = up.rolling(rsi_window_length_1).mean()
    roll_down_14 = down.abs().rolling(rsi_window_length_1).mean()
    # Calculate the RSI based on SMA
    RS_14 = roll_up_14 / roll_down_14
    stock_data["14-day RSI"] = 100.0 - (100.0 / (1.0 + RS_14))
    #plot figure
    fig_4 = make_subplots(rows = 5, cols = 1, shared_xaxes=True, vertical_spacing=0.00625, specs = [[{"rowspan": 3}], [None], [None], [{}], [{}]])
    #add traces - Candlestick, 20-day MA, 40-day MA, RSI, and Volume
    fig_4.add_trace(
        go.Candlestick(
            x = stock_data.index, open = stock_data["Open"], high = stock_data["High"], low = stock_data["Low"], close = stock_data["Close"],
            name = "Price", increasing_line_color = "green", increasing = dict(line = dict(color = "black"))), 
            row = 1, col = 1)
    fig_4.update_layout(xaxis_rangeslider_visible = False)  #disable slider
    fig_4.add_trace(
    go.Scatter(
        x = stock_data.index, y = stock_data["5-day MA"], mode = "lines", name = "5-day MA", line = dict(color = "blue")))
    fig_4.add_trace(
    go.Scatter(
        x = stock_data.index, y = stock_data["10-day MA"], mode = "lines", name = "10-day MA", line = dict(color = "orange")))
    #add RSI
    fig_4.add_scatter(
        x = stock_data.index, y = stock_data["14-day RSI"], mode = "lines", name = "14-day RSI", line = dict(color = "purple"), row = 4, col = 1)
    fig_4.update_layout(
        shapes = [
            dict(type="line", xref="x2", yref="y2", line = dict(color = "green", dash = "dot"),
                x0=min(stock_data.index), y0=30, x1=max(stock_data.index), y1=30),
            dict(type="line", xref="x2", yref='y2', line = dict(color = "red", dash = "dot"),
                x0=min(stock_data.index), y0=70, x1=max(stock_data.index), y1=70)])
    #add volume
    fig_4.add_bar(
        x = stock_data.index, y = stock_data["Volume"], name = "Volume", row = 5, col = 1)
    #add y-axis titles
    fig_4.update_yaxes(title_text="($ USD)", row=1, col=1)
    fig_4.update_yaxes(title_text="(%)", row=4, col=1)

    fig_4.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke",
    margin=dict(l=5, t=40, b=20), title_text= f"Stock: {stock_selected.upper()}")
    fig_4.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_4.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    
    return fig_4

#difference in 5 and 10-day MAs plot
@app.callback(
    Output(component_id = "delta_ma_2", component_property = "figure"),
    [Input(component_id = "search_box_2", component_property = "value"),
    Input(component_id = "start_date_2", component_property = "value"),
    Input(component_id = "end_date_2", component_property = "value"),]
)
def update_delta_MA_graph_2(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate the moving averages (MAs) - weekly trader so I'll focus on 5, 10 and 20 day MAs
    stock_data["5-day MA"] = round(stock_data['Adj Close'].rolling(window = 5).mean(), 6)
    stock_data["10-day MA"] = round(stock_data['Adj Close'].rolling(window = 10).mean(), 6)
    stock_data["20-day MA"] = round(stock_data['Adj Close'].rolling(window = 20).mean(), 6)
    #weekly delta MA insight
    weekly_data = stock_data.tail(14)
    weekly_data["5 - 15 day MA (%)"] = (weekly_data["5-day MA"] - weekly_data["10-day MA"])*100/weekly_data["5-day MA"]
    #Change in Moving Average Crossover (Difference between 2 MAs)
    fig_5 = go.Figure()
    fig_5.add_trace(
        go.Scatter(
            x = weekly_data.index, y = weekly_data["5 - 15 day MA (%)"], mode = "lines", name = "Δ MA (%)", line = dict(color = "royalblue")))
    fig_5.update_layout(
        shapes = [
            dict(type = "line", line = dict(color = "green", dash = "dot"), 
            x0 = min(weekly_data.index), y0 = 0.1, x1 = max(weekly_data.index), y1 = 0.1),
            dict(type = "line", line = dict(color = "red", dash = "dot"), 
            x0 = min(weekly_data.index), y0 = 0.5, x1 = max(weekly_data.index), y1 = 0.5),
            dict(type = "line", line = dict(color = "lightgrey", width = 0.05), 
            x0 = min(weekly_data.index), y0 = 0, x1 = max(weekly_data.index), y1 = 0)])
    # Create scatter trace of text labels
    fig_5.add_trace(go.Scatter(
        x=[min(weekly_data.index)+((max(weekly_data.index)-min(weekly_data.index))/2), min(weekly_data.index)+((max(weekly_data.index)-min(weekly_data.index))/2)],
        y=[0.05, 0.45], text=["Min Buy Signal", "Max Sell Signal"], mode="text", showlegend=False))
    fig_5.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke", title_x=0.5,
    margin=dict(l=5, t=40, b=20), title_text= f"2-Week Δ 5 & 10-day MAs (%), {stock_selected.upper()}")
    fig_5.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_5.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    
    return fig_5

#5 and 14-day RSIs plot
@app.callback(
    Output(component_id = "rsi_2", component_property = "figure"),
    [Input(component_id = "search_box_2", component_property = "value"),
    Input(component_id = "start_date_2", component_property = "value"),
    Input(component_id = "end_date_2", component_property = "value"),]
)
def update_delta_RSI_graph_2(stock_selected, start_date, end_date):
    start = datetime.strptime(start_date, '%m/%d/%Y')
    end = datetime.strptime(end_date, '%m/%d/%Y')
    stock_data = web.DataReader(stock_selected, 'yahoo', start, end)
    #calculate 14-day RSI
    rsi_window_length_1 = 14  #default is 14 days
    rsi_window_length_2 = 5    #for a much more robust response to price changes
    close = stock_data["Adj Close"]
    #Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the SMA - simple moving average
    #using 14 day interval
    roll_up_14 = up.rolling(rsi_window_length_1).mean()
    roll_down_14 = down.abs().rolling(rsi_window_length_1).mean()
    #using 5 day interval
    roll_up_5 = up.rolling(rsi_window_length_2).mean()
    roll_down_5 = down.abs().rolling(rsi_window_length_2).mean()
    # Calculate the RSI based on SMA
    RS_14 = roll_up_14 / roll_down_14
    RS_5 = roll_up_5 / roll_down_5
    stock_data["14-day RSI"] = 100.0 - (100.0 / (1.0 + RS_14))
    stock_data["5-day RSI"] = 100.0 - (100.0 / (1.0 + RS_5))
    #get RSI data for one month
    monthly_rsi = stock_data[["14-day RSI", "5-day RSI"]].tail(31)
    fig_6 = go.Figure()
    #add the 2 RSI values (5 and 14-day)
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"], line = {"color": "purple"}, name = "14-day")
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"], line = {"color": "blue"}, name = "5-day")
    #color change for intervals
    #14-day RSI
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"].where(monthly_rsi["14-day RSI"] <= 30), line = {"color": "green"}, showlegend=False)
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["14-day RSI"].where(monthly_rsi["14-day RSI"] >= 70), line = {"color": "red"}, showlegend=False)
    #5-day interval
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"].where(monthly_rsi["5-day RSI"] <= 30), line = {"color": "green"}, showlegend=False)
    fig_6.add_scattergl(x = monthly_rsi.index, y = monthly_rsi["5-day RSI"].where(monthly_rsi["5-day RSI"] >= 70), line = {"color": "red"}, showlegend=False)

    #thresholds: 70 % overbought, 30 % oversold
    fig_6.update_layout(
        shapes = [
            dict(type = "line", line = dict(color = "green", dash = "dot"), 
            x0 = min(monthly_rsi.index), y0 = 30, x1 = max(monthly_rsi.index), y1 = 30),
            dict(type = "line", line = dict(color = "red", dash = "dot"), 
            x0 = min(monthly_rsi.index), y0 = 70, x1 = max(monthly_rsi.index), y1 = 70)
        ]
    )
    # Create scatter trace of text labels
    fig_6.add_trace(go.Scatter(
        x=[min(monthly_rsi.index)+timedelta(days=5), min(monthly_rsi.index)+timedelta(days=5)],
        y=[25, 65],
        text=["RSI = 30 %",
            "RSI = 70 %"],
        mode="text",
        showlegend=False
    ))

    fig_6.update_layout(plot_bgcolor = "white", paper_bgcolor = "whitesmoke", title_x=0.5,
    margin=dict(l=5, t=40, b=20), title_text= f"Monthly 5 & 14-day RSI (%), {stock_selected.upper()}")
    fig_6.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, 
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")
    fig_6.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
    showgrid=True, gridwidth=1, gridcolor='lightgrey', zerolinecolor='black', zerolinewidth=1, ticks="outside")

    return fig_6
#----------------------------------------------------------------------------------------------
"""Run Application - Local Host"""
#----------------------------------------------------------------------------------------------
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1')
