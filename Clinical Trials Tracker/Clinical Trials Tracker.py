
"""Clinical Trial Tracker"""

"""Create A Drug Trial Tracker on ClinicalTrials.gov"""

"""Create a tool that can report various metrics on drugs or interventions in trial or development on ClinicalTrials.gov"""

#----------------------------------------------------------------------------------------------
#import packages
#----------------------------------------------------------------------------------------------
#landing page: https://clinicaltrials.gov/
import pandas as pd 
import numpy as np 
import plotly.express as px
from datetime import datetime

import bs4
from bs4 import BeautifulSoup as soup
import urllib
from urllib.request import urlopen as uReq
import webbrowser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table


#----------------------------------------------------------------------------------------------
"""Define Search Engine"""
#----------------------------------------------------------------------------------------------
path_start = 'https://clinicaltrials.gov/ct2/results/download_fields?cond='
disease = input('Please input disease of interest: ')
disease_url = disease.replace(" ", "+")
other_term = input("Please input any other key word: ")
other_term_url = other_term.replace(" ", "+")
num_records = input('Enter number of records to analyze, 10/ 100/ 1000/ 10000 (maximum): ')
download_format = input('Enter dowload format (plain, csv, tsv, xml, pdf). We recommend csv: ')
path_end = '&term='+other_term_url+'&down_count='+num_records+'&down_flds=all&down_fmt='+download_format

url = path_start+disease_url+path_end
clinical_trials_data = pd.read_csv(url)
#----------------------------------------------------------------------------------------------
"""Clean Data"""
#----------------------------------------------------------------------------------------------
#insights needed include:
insight_columns = ['Status', 'Study Results','Conditions', 'Interventions', 'Gender', 'Age', 'Phases', 
'Enrollment', 'Study Type', 'Primary Completion Date', 'Completion Date', 'Locations', 'URL']

final_data = clinical_trials_data[insight_columns]

#clean data: conditions, interventions, study type, locations, completion date
final_data["Conditions"] = final_data["Conditions"].str[:18]
final_data["Interventions"] = final_data["Interventions"].str.split(':').str[0]
final_data["Study Type"] = final_data["Study Type"].replace("Expanded Access:Individual Patients", "Expanded Access")
final_data["Locations"] = final_data["Locations"].str.split(",").str[-1]
final_data["Completion Date"] = pd.to_datetime(final_data["Completion Date"])

#create graphs and tables that do not need a drop down
#Locations/ Clinical Trials Map
maps_data = pd.DataFrame(final_data.Locations.value_counts())
map_figure = px.scatter_geo(data_frame = maps_data, locations = maps_data.index, locationmode = "country names", color = maps_data.index,
size = "Locations", template="plotly_dark", labels={"Locations": "Number of Studies"},
title = f"Clinical Trials Global Map").update_layout(title_x = 0.42, title_y = 0.83)

#Summary Table 
summary_table = {
    "Category": ["Number of Studies", "Country with most studies", "Largest Clinical Trial", "Smallest Clinical Trial (Recruiting)",
    "Most Common Intervention", "Most Common Study Type", "Latest Completion Date"], 
    "Insight": [final_data.shape[0], final_data.Locations.value_counts().keys().tolist()[0]+f", ({final_data.Locations.value_counts().max()})",
    round(final_data.Enrollment.max()), round(final_data.Enrollment[final_data.Enrollment != 0].min()), 
    final_data.Interventions.value_counts().keys().tolist()[0]+f", ({final_data.Interventions.value_counts().max()})",
    final_data['Study Type'].value_counts().keys().tolist()[0]+f", ({final_data['Study Type'].value_counts().max()})",
    final_data["Completion Date"].max().strftime('%m-%d-%Y')]
}

summary = pd.DataFrame(
    summary_table,
    columns = ["Category", "Insight"])


def application():
    #----------------------------------------------------------------------------------------------
    """Initialize Dashboard Application"""
    #----------------------------------------------------------------------------------------------
    app = dash.Dash(__name__)

    #----------------------------------------------------------------------------------------------
    """Application Layout"""
    #----------------------------------------------------------------------------------------------
    app.layout = html.Div(children = [
        html.Div(html.H1("ClinicalTrials.gov Insights Tracker", 
                                    style = {
                                    "color": "lightgrey", "font-size": "35px", "font-family": "Courier", "font-weight": "bold",
                                    "text-align": "center"}), 
                                    style = {
                                        "display": "inline-block", "width": "100%", "background-color": "#198805"}),

        html.Div(html.H3(f"Disease Of Interest: {disease.capitalize()}, Other Keyword: {other_term.capitalize()}", 
                                    style = {
                                    "color": "black", "font-size": "20px", "font-family": "Courier New", "text-align": "center"}),
                                    style = {
                                        "display": "inline-block", "width": "100%", "background-color": "#458f39"  
                                    }),

        html.Div(children = [
            html.Div([
                html.Br(),
                html.Br(),

                html.P("Select Categorical Insight", 
                    style = {
                        "font-size": "20px", "text-align": "center", "color": "black", "font-family": "Courier New"}),
                
                html.Br(),
                html.Br(),

                html.Div(dcc.Dropdown(id = "categorical_dropdown", 
                    options = [
                                {"label": "Status", "value": "Status"},
                                {"label": "Study Results", "value": "Study Results"},
                                {"label": "Conditions", "value": "Conditions"},
                                {"label": "Interventions", "value": "Interventions"},
                                {"label": "Gender", "value": "Gender"},
                                {"label": "Phases", "value": "Phases"},
                                {"label": "Study Type", "value": "Study Type"}
                    ],
                    multi = False,
                    value = "Status",
                    style = {"font-size": "15px", "font-family": "Courier", "text-align": "center", "color": "black", "background-color":"#0f0f0f"}))],
                className = "two columns", style = {"background-color": "#a0f296", "height": "350px", "display": "inline-block"}),

            html.Div(dcc.Graph(id = "categorical_plots", figure = [], style = {"height": "350px"}), style = {"display": "inline-block"}, className = "six columns"),
            
            html.Div([
                html.Br(),

                html.P("Clinical Trials Summary", 
                style = {
                    "font-size": "18px", "text-align": "center", "color": "black", "font-family": "Courier New", "text-decoration": "underline"}),
                    
                html.Br(),

                html.Div(dash_table.DataTable(
                    id = "summary_table",
                    columns=[{"name": i, "id": i} for i in summary.columns],
                    data = summary.to_dict("records"),
                    style_cell = {'textAlign': 'left', "color": "black"},
                    style_header = {"font-weight": "bold", 'border': '2px solid grey', "background-color": "#8acf82", "text-align": "center"},
                    style_data = {'border': '2px solid grey', "background-color": "#8acf82"}))],
                style = {
                    "display": "inline-block", "background-color": "#8acf82", "height": "350px"}, className = "four columns")], 
            
            className = "row"),

        html.Div(children = [
            html.Div([
                html.Br(),
                html.Br(),

                html.P("Select Non-Categorical Insight", 
                    style = {
                        "font-size": "20px", "text-align": "center", "color": "black", "font-family": "Courier New"}),

                html.Br(),
                html.Br(),

                html.Div(dcc.Dropdown(id = "non_categorical_dropdown",
                    options = [
                                {"label": "Enrollment", "value": "Enrollment"},
                                {"label": "Completion Date", "value": "Completion Date"}
                    ],
                    multi = False,
                    value = "Enrollment",
                    style = {"font-size": "15px", "font-family": "Courier", "text-align": "center", "color": "black", "background-color":"#0f0f0f"}))],
                className = "two columns", style = {"background-color": "#bcf0b6", "height": "350px", "display": "inline-block"}),

            html.Div(dcc.Graph(id = "non_categorical_plots", figure = [], style = {"height": "350px"}), style = {"display": "inline-block"}, className = "five columns"),
            
            html.Div(dcc.Graph(id = "map", figure = map_figure, style = {"height": "350px"}), style = {"display": "inline-block"}, className = "five columns")], 
            
            className = "row")

    ], style = {"background-color": "#0f0f0f"}, className = "twelve colummns")

    #----------------------------------------------------------------------------------------------
    """Define Callbacks"""
    #----------------------------------------------------------------------------------------------
    #categorical plots
    @app.callback(
        Output(component_id = "categorical_plots", component_property = "figure"),
        [Input(component_id = "categorical_dropdown", component_property = "value")]
    )
    def update_frequency_plots(category_selected):
        df = final_data.copy()
        dff = df[category_selected].value_counts()

        if category_selected == "Conditions":
            dff = pd.DataFrame(dff.head(10))

            fig_1 = px.bar(data_frame = dff, x = dff[category_selected], y = dff.index, orientation='h', color = category_selected, template = "plotly_dark",
            labels = {category_selected: "Number of Studies", "y": category_selected}, title = f"Clinical Trials, Top 10 {category_selected}, Total Studies = {df.shape[0]}").update_layout(title_x=0.5, yaxis=dict(autorange="reversed"), title_y = 0.83)

        else:
            dff = pd.DataFrame(dff)
            fig_1 = px.bar(data_frame = dff, x = dff[category_selected], y = dff.index, orientation='h', color = category_selected, template = "plotly_dark",
            labels = {category_selected: "Number of Studies", "y": category_selected}, title = f"Clinical Trials, {category_selected} Distribution, Total Studies = {df.shape[0]}").update_layout(title_x=0.5, yaxis=dict(autorange="reversed"), title_y = 0.83)

        return fig_1 

    #map and time series
    @app.callback(
        Output(component_id = "non_categorical_plots", component_property = "figure"),
        [Input(component_id = "non_categorical_dropdown", component_property = "value")]
    )

    def update_map_time_series_plot(cat_selected):
        xf = final_data.copy()

        if cat_selected == "Enrollment":
            fig_2 = px.histogram(xf, x = "Enrollment", template="plotly_dark", color_discrete_sequence=["#198805"],
            title = f"Clinical Trials {cat_selected}, Total Studies = {xf.shape[0]}").update_layout(title_x = 0.5, title_y = 0.83)

        else:
            #define date time for the day
            today = datetime.today()
            today = today.strftime("%Y-%m-%d")
            today = today[0:7]

            xff = pd.DataFrame(xf["Completion Date"].dt.to_period('M').value_counts())
            xff.reset_index(inplace = True)
            xff = xff.rename(columns = {"index": "Date", "Completion Date": "Count"})
            xff.Date = xff.Date.astype(str)
            xff['Completed'] = np.where(xff['Date'] < today, 'Completed', 'Incomplete')
            xff = xff[xff.Date < '2060-01']   #invalidate studies with a date above 2060-01

            fig_2 = px.scatter(xff, x = "Date", y = "Count", size = "Count", color = "Completed",
            template = "plotly_dark", title = f"Clinical Trials {cat_selected}, Total Studies = {xff.Count.sum()}").update_layout(title_x = 0.5, title_y = 0.83)

        return fig_2
    #----------------------------------------------------------------------------------------------
    """Run Application"""
    #----------------------------------------------------------------------------------------------
    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })
    if __name__ == '__main__':
        app.run_server(port=8000, host='127.0.0.1')

#run application
application()


