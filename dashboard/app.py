import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import requests
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Fetch summary statistics from API
summary_response = requests.get("http://127.0.0.1:5000/summary").json()
total_transactions = summary_response.get("total_transactions", "N/A")
total_fraud_cases = summary_response.get("fraud_cases", "N/A")
fraud_percentage = summary_response.get("fraud_percentage", "N/A")

# Fetch fraud trends
trends_response = requests.get("http://127.0.0.1:5000/fraud_trends").json()
fraud_trends_df = pd.DataFrame(
    trends_response) if trends_response else pd.DataFrame()

# Fetch fraud data by device and browser
fraud_by_device_browser = requests.get(
    "http://127.0.0.1:5000/fraud_by_device_browser").json()
fraud_by_device = fraud_by_device_browser.get("fraud_by_device", {})
fraud_by_browser = fraud_by_device_browser.get("fraud_by_browser", {})

device_df = pd.DataFrame(list(fraud_by_device.items()),
                         columns=["device_id", "fraud_cases"])
browser_df = pd.DataFrame(list(fraud_by_browser.items()), columns=[
                          "browser", "fraud_cases"])

# Dashboard layout
app.layout = dbc.Container([
    html.H1("Fraud Detection Dashboard"),
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4("Total Transactions"),
            html.H2(f"{total_transactions}"),
        ], body=True)),
        dbc.Col(dbc.Card([
            html.H4("Total Fraud Cases"),
            html.H2(f"{total_fraud_cases}"),
        ], body=True)),
        dbc.Col(dbc.Card([
            html.H4("Fraud Percentage"),
            html.H2(f"{fraud_percentage}%"),
        ], body=True)),
    ], className="mb-4"),

    dcc.Graph(id="fraud-trend", figure=px.line(fraud_trends_df,
              x="date", y="fraud_cases", title="Fraud Cases Over Time")),
    dcc.Graph(id="fraud-by-device", figure=px.bar(device_df,
              x="device_id", y="fraud_cases", title="Fraud Cases by Device")),
    dcc.Graph(id="fraud-by-browser", figure=px.bar(browser_df,
              x="browser", y="fraud_cases", title="Fraud Cases by Browser")),

    html.Hr(),
    html.H3("Fraud Prediction"),
    dbc.Row([
        dbc.Col(dcc.Input(id=col, type="number", placeholder=col.replace("_", " ").title())) for col in [
            "user_id", "transaction_frequency", "signup_time", "purchase_time",
            "velocity_check", "purchase_hour", "purchase_weekday", "purchase_value",
            "device_id", "source", "browser", "sex", "age", "ip_address", "country"
        ]
    ], className="mb-2"),

    dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3"),
    html.Div(id="prediction-result", className="mt-3"),
])

# Prediction callback


@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    [State(col, "value") for col in [
        "user_id", "transaction_frequency", "signup_time", "purchase_time",
        "velocity_check", "purchase_hour", "purchase_weekday", "purchase_value",
        "device_id", "source", "browser", "sex", "age", "ip_address", "country"
    ]]
)
def get_prediction(n_clicks, *values):
    if n_clicks:
        response = requests.post(
            "http://127.0.0.1:5000/predict", json={"features": values})
        prediction = response.json().get("prediction", "Error")
        return html.H4(f"Predicted Fraud Status: {'Fraud' if prediction == 1 else 'Not Fraud'}")
    return ""


if __name__ == "__main__":
    app.run(debug=True, port=8000)
