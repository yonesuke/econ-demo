import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import numpy as np
from scipy.stats import dirichlet

yf.pdr_override()

@st.cache_data
def get_data(symbol, start, end):
    return web.get_data_yahoo(tickers=symbol,start=start,end=end)

# 銘柄名入力欄
symbols = st.sidebar.multiselect(
    'Company',
    (
        'AAPL',
        'MSFT',
        'GOOG',
        'AMZN',
        'BRK-A',
    ),
    placeholder='Select ticker ...'
)

start = st.sidebar.date_input('Start Date', dt.date(2016, 3, 1))
end = st.sidebar.date_input('End Date', dt.date.today())

n_montecarlo = st.sidebar.slider('Number of Samples', 10, 10000)

st.title('Portfolio Selction Problem')

df = pd.DataFrame(
    {symbol: get_data(symbol, start, end)['Adj Close'] for symbol in symbols}
)
df.index = pd.to_datetime(df.index).date

st.dataframe(df, use_container_width=True)

df_rtn = df.pct_change().dropna()
ser_mean = df_rtn.mean().apply(lambda x: (1 + x)**250 - 1)
ser_std = df_rtn.std().apply(lambda x: x * np.sqrt(250))

df_mean_std = pd.DataFrame({'mean': ser_mean, 'std': ser_std})

if len(symbols) > 1:
    weights = dirichlet.rvs([1.0]*len(symbols), n_montecarlo)
    # weights[i][j]がiサンプル目のj番目のsymbolのリターンに掛け算をする
    # つまり、iサンプル目のリターンは、weights[i][0] * df_rtn.iloc[:, 0] + weights[i][1] * df_rtn.iloc[:, 1] + ...
    # となる
    # これをn_montecarlo回繰り返す
    rtn_samples = weights @ df_rtn.values.transpose()
    mean_samples = (1 + rtn_samples.mean(axis=-1))**250 - 1
    std_samples = rtn_samples.std(axis=-1) * np.sqrt(250)
    sharpe_ratio_samples = mean_samples / std_samples
    idx_max = np.argmax(sharpe_ratio_samples)
    slope = sharpe_ratio_samples[idx_max]
    weight_optimal = weights[idx_max]
    df_mean_std = pd.DataFrame({'mean': ser_mean, 'std': ser_std, 'optimal weight': weight_optimal})

pd.options.display.float_format = '{:.2f}'.format
st.dataframe(
    df_mean_std * 100
)

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
fig.update_xaxes(title='Standard Deviation')
fig.update_yaxes(title='Expected Return')
for symbol in symbols:
    fig.add_trace(
        go.Scatter(
            x=df_mean_std.loc[[symbol], 'std'] * 100,
            y=df_mean_std.loc[[symbol], 'mean'] * 100,
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
            name='Efficient Frontier', showlegend=True,
            mode='lines'
        )
    )
    fig.update_xaxes(range=[0.98 * x_min, 1.02 * x_max])
    fig.update_yaxes(range=[0.98 * y_min, 1.02 * y_max])
st.plotly_chart(fig)
