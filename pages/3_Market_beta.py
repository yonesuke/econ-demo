import streamlit as st
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import numpy as np

pd.options.plotting.backend = "plotly"

yf.pdr_override()

url_topix_companies = 'https://www.jpx.co.jp/markets/indices/topix/tvdivq00000030ne-att/topixweight_j.csv'

@st.cache_data
def get_topix_companies():
    df_topix_companies = pd.read_csv(url_topix_companies, encoding='shift-jis', dtype={'コード': str})
    df_topix_companies = df_topix_companies[:df_topix_companies[df_topix_companies.iloc[:, 1].isna()].index[0]]
    return df_topix_companies

@st.cache_data
def get_prc(ticker, start, end):
    return web.get_data_yahoo(f'{ticker}.T', start=start, end=end)['Close']

@st.cache_data
def get_tpx(start, end):
    return web.get_data_stooq('^TPX', start, end)['Close'].sort_index()

# get data
df_topix_companies = get_topix_companies()
company_all = df_topix_companies.apply(lambda ser: ser[1]+f' ({ser[2]})', axis=1).values
ticker_all = df_topix_companies.iloc[:, 2].values
dict_company2ticker = dict(zip(company_all, ticker_all))

# streamlit構成
st.title('Market beta')
# sidebar
# 銘柄名入力欄
companies = st.sidebar.multiselect('Companies', company_all, ['トヨタ自動車 (7203)', '野村ホールディングス (8604)'])
tickers = [dict_company2ticker[company] for company in companies]
st.sidebar.caption(f'List of companies retrieved from [JPX page](https://www.jpx.co.jp/markets/indices/topix/#heading_3)')
# frequency
freq = st.sidebar.selectbox(
    'Data Frequency',
    ('Daily', 'Weekly', 'Monthly')
)
# start date: two years ago from now
dt_start = st.sidebar.date_input('Start Date', dt.date.today() - dt.timedelta(days=730))
# end date: today
dt_end = st.sidebar.date_input('End Date', dt.date.today())
# Risk Free Rate
rate_free = st.sidebar.number_input('Risk Free Rate [%]', value=0.0)

# get data
_df_prc = pd.DataFrame(
    {
        companies[i]: get_prc(tickers[i], dt_start, dt_end) for i in range(len(companies))
    }
)
_ser_tpx = get_tpx(dt_start, dt_end)[_df_prc.index]
if freq == 'Daily':
    df_prc = _df_prc.fillna(method='ffill')
    ser_tpx = _ser_tpx.fillna(method='ffill')
elif freq == 'Weekly':
    df_prc = _df_prc.resample('W').last().fillna(method='ffill')
    ser_tpx = _ser_tpx.resample('W').last().fillna(method='ffill')
elif freq == 'Monthly':
    df_prc = _df_prc.resample('M').last().fillna(method='ffill')
    ser_tpx = _ser_tpx.resample('M').last().fillna(method='ffill')

# plot
tab1, tab2, tab3 = st.tabs(['Close Price', 'Indexing the start date as 100', 'Log Return'])
with tab1:
    fig = go.Figure()
    for company in companies:
        fig.add_trace(
            go.Scatter(
                x=df_prc.index, y=df_prc[company], name=company
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ser_tpx.index, y=ser_tpx, name='TOPIX'
        )
    )
    fig.update_layout(
        title=f'Close Price ({freq})',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend_title='Company'
    )
    st.plotly_chart(fig)
with tab2:
    fig = go.Figure()
    for company in companies:
        fig.add_trace(
            go.Scatter(
                x=df_prc.index, y=df_prc[company]/df_prc[company][0]*100, name=company
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ser_tpx.index, y=ser_tpx/ser_tpx[0]*100, name='TOPIX'
        )
    )
    fig.update_layout(
        title=f'Indexing the start date as 100 ({freq})',
        xaxis_title='Date',
        yaxis_title='Indexing the start date as 100',
        legend_title='Company'
    )
    st.plotly_chart(fig)
with tab3:
    fig = go.Figure()
    for company in companies:
        fig.add_trace(
            go.Scatter(
                x=df_prc.index, y=np.log(df_prc[company]).diff()*100, name=company
            )
        )
    fig.add_trace(
        go.Scatter(
            x=ser_tpx.index, y=np.log(ser_tpx).diff()*100, name='TOPIX'
        )
    )
    fig.update_layout(
        title=f'Log Return ({freq})',
        xaxis_title='Date',
        yaxis_title='Log Return [%]',
        legend_title='Company'
    )
    st.plotly_chart(fig)

def calc_beta(company):
    x, y = (ser_tpx.pct_change()*100).dropna().values, (df_prc[company].pct_change()*100).dropna().values
    beta, _ = np.polyfit(x, y, 1)
    return beta

st.dataframe(
    pd.DataFrame(
        index=['beta'],
        data={company: calc_beta(company) for company in companies}
    )
)

st.markdown(
f"""
## Market Return - Return Plot ({freq})
"""
)

tabs = st.tabs(companies)
for i in range(len(companies)):
    company = companies[i]
    with tabs[i]:
        fig = go.Figure()
        x, y = (ser_tpx.pct_change()*100).dropna().values, (df_prc[company].pct_change()*100).dropna().values
        fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    name=company, mode='markers'
                )
            )
        # get ols line with numpy
        slope, intercept = np.polyfit(x, y, 1)
        fig.add_trace(
            go.Scatter(
                x=x, y=slope*x+intercept,
                name=rf'OLS, beta: {slope:.2f}', mode='lines'
            )
        )
        fig.update_layout(
            xaxis_title='Market Return [%]',
            yaxis_title='Return [%]'
        )
        st.plotly_chart(fig)
