import pandas_datareader as web
import datetime as dt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st

st.title('Crypto Forecast App')

coin = ('BTC', 'ETH',  'USDT', 'USDC', 'BNB', 'ADA', 'XRP', 'BUSD', 'SOL', 'DOGE', 'DOT', 'WBTC','DAI', 'LTC', 'AVAX', 'UNI', 'TUSD', 'FIL', 'XMR')
selected_coin = st.selectbox('Select Coin for prediction', coin)


def load_data_yahoo(coin):
    return web.DataReader(coin, 'yahoo', dt.datetime(2016, 1, 1), dt.datetime.now())


FORECAST_IN_DAYS = st.select_slider('Select a Date Range to Predict', options=[1, 7, 30, 60, 90])

data_load_state = st.text('Loading data...')
data = load_data_yahoo(selected_coin + "-USD")
data.reset_index(inplace=True)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.head())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="coin_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="coin_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future_dates = model.make_future_dataframe(periods=FORECAST_IN_DAYS)
prediction = model.predict(future_dates)

# Show and plot forecast
st.subheader('Forecast data')
st.write(prediction.head())

st.write(f'Forecast plot for {FORECAST_IN_DAYS} days')
fig1 = plot_plotly(model,prediction)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = model.plot_components(prediction)
st.write(fig2)