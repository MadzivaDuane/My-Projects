"""Simple Tool To Visually Track COVID-19 New Cases, New Deaths, Total Cases and Total Deaths in Sub Saharan Africa

3 main goals:
1. Create a interactive visual dashboard for COVID information  as of yesterday (new cases, new deaths, total cases, total deaths)
2. Show trends of COVID-19 since earliest case in Africa
3. Create an interactive animation to show development of total cases and deaths across Sub Saharan Africa
"""
import pandas as pd 
import numpy as np 
import random
import matplotlib
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import plotly.express as px
from functools import partial

#load data
#Data Source: Data World - https://data.world/markmarkoh/coronavirus-data
#Full Data url: https://query.data.world/s/3gmtix24pttvfrd5ocdeddmxrihnj7
covid_data = pd.read_csv('https://query.data.world/s/3gmtix24pttvfrd5ocdeddmxrihnj7') 
covid_data.head()

#Sub Sahran Africa Country Codes
path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data Science Projects/COVID 19 - Sub Saharan Africa/data/"
country_codes = pd.read_csv(path+"country_codes.csv"); country_codes.head()

#data for Sub Saharan Africa, and preprocessing 
sub_sahara_data = pd.DataFrame(columns = ["date", "location", "new_cases", "new_deaths", "total_cases", "total_deaths"])
for country in np.array(country_codes.Country):
    index = covid_data.loc[covid_data["location"] == country]
    sub_sahara_data = sub_sahara_data.append(index)
sub_sahara_data.shape 

#merge datasets to include codes
final_data = pd.merge(sub_sahara_data, country_codes, left_on="location", right_on ="Country")
final_data.shape  
final_data.isnull().sum()  #check for missing data
#convert to correct datatypes, and rename columns
final_data[["new_cases", "new_deaths", "total_cases", "total_deaths"]] = final_data[["new_cases", "new_deaths", "total_cases", "total_deaths"]].astype(np.int64)
final_data = final_data.rename(columns={'date': 'Date', 'new_cases': 'New Cases','new_deaths': 'New Deaths','total_cases': 'Total Cases','total_deaths': 'Total Deaths', 'Country_Code': 'Country Code'})
final_data.dtypes

#=====================================================================================================
#Goal 1: Visual Dashboard of COVID-19 as of Yesterday
#----------------------------------------------------------------------------------------------------
#get yesterdays date 
yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
#dataframe with yesterdays information for sub saharan africa
df_yesterday = final_data[final_data.Date == yesterday]; df_yesterday.head()
#----------------------------------------------------------------------------------------------------
"""New Cases"""
fig_new_cases = px.scatter_geo(
    df_yesterday, 
    locations='Country Code', color='Country', hover_name='Country', size='New Cases',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 New Cases as of {yesterday}')
fig_new_cases.show()
#----------------------------------------------------------------------------------------------------
"""New Deaths"""
fig_new_deaths = px.scatter_geo(
    df_yesterday, 
    locations='Country Code', color='Country', hover_name='Country', size='New Deaths',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 New Deaths as of {yesterday}')
fig_new_deaths.show()
#----------------------------------------------------------------------------------------------------
"""Total Cases"""
fig_total_cases = px.scatter_geo(
    df_yesterday, 
    locations='Country Code', color='Country', hover_name='Country', size='Total Cases',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 Total Cases as of {yesterday}')
fig_total_cases.show()
#----------------------------------------------------------------------------------------------------
"""Total Deaths"""
fig_total_deaths = px.scatter_geo(
    df_yesterday, 
    locations='Country Code', color='Country', hover_name='Country', size='Total Deaths',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 Total Deaths as of {yesterday}')
fig_total_deaths.show()

#====================================================================================================
#Goal 2: COVID-19 Time Trend Analysis
#----------------------------------------------------------------------------------------------------
"""New Cases"""
#New Cases per Day Sub-Saharan Africa
daily_new_cases = final_data.groupby('Date', as_index=False).agg({"New Cases": "sum"})
fig_daily_new_cases = px.line(
    daily_new_cases, 
    x = "Date", y = "New Cases", title="Sub Saharan Africa COVID-19 Daily New Cases")
fig_daily_new_cases.show()
#----------------------------------------------------------------------------------------------------
"""New Deaths"""
daily_new_deaths = final_data.groupby('Date', as_index=False).agg({"New Deaths": "sum"})
fig_daily_new_deaths = px.line(
    daily_new_deaths, 
    x = "Date", y = "New Deaths", title="Sub Saharan Africa COVID-19 Daily New Deaths")
fig_daily_new_deaths.show()
#----------------------------------------------------------------------------------------------------
"""Total Cases"""
time_trend_total_cases = px.line(
    final_data, 
    x="Date", y="Total Cases", 
    color='Country', hover_name='Country', title="Total Cases - Daily Trend")
time_trend_total_cases.show()
#----------------------------------------------------------------------------------------------------
"""Total Deaths"""
time_trend_total_deaths = px.line(
    final_data, 
    x="Date", y="Total Deaths", 
    color='Country', hover_name='Country', title="Total COVID-19 Deaths - Daily Trend")
time_trend_total_deaths.show()

#====================================================================================================
#Goal 3: Animation showing growth of COVID-19 total cases and deaths in Sub Saharan Africa over time
#----------------------------------------------------------------------------------------------------
#create new dataset and reformat date information
animation_data =  final_data
to_datetime_fmt = partial(pd.to_datetime, format='%Y-%m-%d')
animation_data['Date'] = animation_data['Date'].apply(to_datetime_fmt)
animation_data['Date'] = animation_data.Date.dt.strftime('%Y%m%d')
animation_data = animation_data.sort_values(by=['Date'])
animation_data = animation_data.dropna()
#add continent name for better animation
animation_data["Continent"] = "Africa"
#check for missing values to avoid complications
animation_data.isnull().sum(); animation_data.head()  #expect different starting times for COVID, some coutries had cases earlier
#----------------------------------------------------------------------------------------------------
"""Total Cases"""
fig_animation_total_cases = px.scatter_geo(
    animation_data, 
    locations='Country Code', color='Continent', hover_name='Country', size='Total Cases',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 Total Cases over time',
    animation_frame="Date")
fig_animation_total_cases.show()
#----------------------------------------------------------------------------------------------------
"""Total Deaths"""
fig_animation_total_deaths = px.scatter_geo(
    animation_data, 
    locations='Country Code', color='Continent', hover_name='Country', size='Total Deaths',
    scope = "africa", projection="equirectangular", title=f'Sub Saharan Africa COVID-19 Total Deaths over time',
    animation_frame="Date")
fig_animation_total_deaths.show()


