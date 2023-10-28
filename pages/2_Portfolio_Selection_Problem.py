import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import numpy as np
from scipy.stats import dirichlet

pd.options.plotting.backend = "plotly"

yf.pdr_override()

@st.cache_data
def get_data(ticker, start, end):
    return web.get_data_yahoo(tickers=ticker,start=start,end=end)

# frequency
freq = st.sidebar.selectbox(
    'Data Frequency',
    ('Daily', 'Weekly', 'Monthly')
)

# 銘柄名入力欄
dict_symbol2ticker= {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google': 'GOOG',
    'Amazon': 'AMZN',
    'Berkshire Hathaway': 'BRK-A',
    'Toyota': '7203.T',
    'Sony': '6758.T',
    'NTT': '9432.T',
    'Nomura': '8604.T',
    'SoftBank': '9984.T'
}
symbols = st.sidebar.multiselect(
    'Company',
    tuple(list(dict_symbol2ticker.keys())),
    placeholder='Select ticker ...',
    default=['Apple', 'Microsoft']
)
tickers = [dict_symbol2ticker[symbol] for symbol in symbols]

start = st.sidebar.date_input('Start Date', dt.date(2016, 3, 1))
end = st.sidebar.date_input('End Date', dt.date.today())

n_montecarlo = st.sidebar.slider('Number of Samples', 10, 10000, value=100)

st.title('Portfolio Selction Problem')

_df = pd.DataFrame(
    {symbols[i]: get_data(tickers[i], start, end)['Close'] for i in range(len(tickers))}
)
if freq == 'Daily':
    multiplier = 250
    df = _df.fillna(method='ffill')
elif freq == 'Weekly':
    multiplier = 52
    df = _df.resample('W').last().fillna(method='ffill')
elif freq == 'Monthly':
    multiplier = 12
    df = _df.resample('M').last().fillna(method='ffill')
# df.index = pd.to_datetime(df.index).date
df.index.name = 'Date'
df.columns.name = 'Company'

tab1, tab2, tab3 = st.tabs(['Close Price', 'Indexing the start date as 100', 'Log Return'])
with tab1:
    fig = df.plot()
    # fig.set_ylabel('Close Price')
    st.plotly_chart(fig)
with tab2:
    fig = df.apply(lambda ser: ser / ser[0] * 100).plot()
    # fig.set_ylabel('Indexing the start date as 100')
    st.plotly_chart(fig)
with tab3:
    fig = df.pct_change().apply(np.log1p).plot()
    st.plotly_chart(fig)
   
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
converted_df = convert_df(df)
st.download_button(
    label="Download data as CSV",
    data=converted_df,
    file_name=f"stock_price_{freq}.csv",
    mime="text/csv",
)

df_rtn = df.pct_change().apply(np.log1p).dropna()
ser_mean = df_rtn.mean().apply(lambda x: (1 + x)**multiplier - 1)
ser_std = df_rtn.std().apply(lambda x: x * np.sqrt(multiplier))

df_mean_std = pd.DataFrame({'Return': ser_mean, 'Risk': ser_std})

if len(symbols) > 1:
    weights = dirichlet.rvs([1.0]*len(symbols), n_montecarlo)
    # weights[i][j]がiサンプル目のj番目のsymbolのリターンに掛け算をする
    # つまり、iサンプル目のリターンは、weights[i][0] * df_rtn.iloc[:, 0] + weights[i][1] * df_rtn.iloc[:, 1] + ...
    # となる
    # これをn_montecarlo回繰り返す
    rtn_samples = weights @ df_rtn.values.transpose()
    mean_samples = (1 + rtn_samples.mean(axis=-1))**multiplier - 1
    std_samples = rtn_samples.std(axis=-1) * np.sqrt(multiplier)
    sharpe_ratio_samples = mean_samples / std_samples
    idx_max = np.argmax(sharpe_ratio_samples)
    idx_minrisk = np.argmin(std_samples)
    slope = sharpe_ratio_samples[idx_max]
    weight_optimal = weights[idx_max]
    weight_minrisk = weights[idx_minrisk]
    df_mean_std = pd.DataFrame({'Return': ser_mean, 'Risk': ser_std, 'Optimal Portfolio': weight_optimal, 'Minimal Risk Portfolio': weight_minrisk})

st.markdown("""
## Mean Variance theory
""")

fig = go.Figure()
# fig title
if len(symbols) < 2:
    fig.update_layout(title='Mean Variance plot')
else:
    fig.update_layout(title=f'Mean Variance plot: Optimal portfolio Sharpe Ratio = {slope:.2f}')
# fig xlabel and ylabel
fig.update_xaxes(title='Risk')
fig.update_yaxes(title='Expected Return')
for symbol in symbols:
    fig.add_trace(
        go.Scatter(
            x=df_mean_std.loc[[symbol], 'Risk'] * 100,
            y=df_mean_std.loc[[symbol], 'Return'] * 100,
            mode='markers', marker_size=10,
            name=symbol, showlegend=True
        )
    )
if len(symbols) > 1:
    fig.add_trace(
        go.Scatter(
            x=std_samples * 100,
            y=mean_samples * 100,
            mode='markers', marker_size=5,
            name='Monte Carlo', showlegend=True,
            opacity=0.4
        )
    )
    # efficient frontierをプロットするために
    # x_min, x_max, y_minを取得する
    traces = fig.data
    x_values = []
    y_values = []
    for trace in traces:
        x_values += trace.x.tolist()
        y_values += trace.y.tolist()
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    # (x_min, y_min)を通り、x_maxまでの範囲で傾きsharpe_ratio_samples[idx_max]の直線をひく
    # これがefficient frontier
    slope = sharpe_ratio_samples[idx_max]
    fig.add_trace(
        go.Scatter(
            x=[0, 100], y=[0, slope * 100],
            name='Tangency Portfolio', showlegend=True,
            mode='lines'
        )
    )
    fig.update_xaxes(range=[0.98 * x_min, 1.02 * x_max])
    fig.update_yaxes(range=[0.98 * y_min, 1.02 * y_max])
st.plotly_chart(fig)

st.dataframe(
    df_mean_std * 100, use_container_width=True
)
