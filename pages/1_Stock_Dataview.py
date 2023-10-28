import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import plotly.graph_objects as go
import datetime as dt

yf.pdr_override()

@st.cache_data
def get_data(symbol, start, end):
    return web.get_data_yahoo(tickers=symbol,start=start,end=end)

# 銘柄名入力欄
symbol = st.sidebar.selectbox(
    'Company',
    (
        'AAPL',
        'MSFT',
        'GOOG',
        'AMZN',
        'META',
        'TSLA',
        'BRK-A',
        'WE'
    ),
    placeholder='Select ticker ...'
)



start = st.sidebar.date_input('Start Date', dt.datetime(2016, 3, 1))
end = st.sidebar.date_input('End Date', dt.datetime.today())

st.title(f'Stock Data of {symbol}')

df = get_data(symbol, start, end)
df.index = pd.to_datetime(df.index).date
df.index.name = 'Date'

# 株価情報表示
st.dataframe(df, use_container_width=True)

fig = go.Figure()
fig.update_layout(title=f'Close price of {symbol}')
fig.add_trace(
    go.Line(
        x=df.index, y=df['Close'],
        name='Close', showlegend=True
    )
)

st.markdown('## Charts')

col1, col2, col3 = st.columns(3)
with col1:
    plot_ma20 = st.checkbox('Moving Average (20 days)')
with col2:
    plot_ma120 = st.checkbox('Moving Average (120 days)')
with col3:
    plot_ma250 = st.checkbox('Moving Average (250 days)')
if plot_ma20:
    fig.add_trace(
        go.Line(
            x=df.index, y=df['Close'].rolling(20).mean(),
            name='MA(20)', showlegend=True
        )
    )
if plot_ma120:
    fig.add_trace(
        go.Line(
            x=df.index, y=df['Close'].rolling(120).mean(),
            name='MA(120)', showlegend=True
        )
    )
if plot_ma250:
    fig.add_trace(
        go.Line(
            x=df.index, y=df['Close'].rolling(250).mean(),
            name='MA(250)', showlegend=True
        )
    )

st.plotly_chart(fig)
