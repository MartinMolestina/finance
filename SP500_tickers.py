import bs4 as bs
import datetime as dt
import pandas as pd
import os
import pickle
import requests
from yahoofinancials import YahooFinancials

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = str(dt.datetime(2010, 1, 1).date())
    end = str(dt.datetime.now().date())
    time_period = 'daily'

    for ticker in tickers:
        try:
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                TKR = YahooFinancials(ticker)
                fin_dict = TKR.get_historical_price_data(start, end, time_period)
                df = pd.DataFrame(fin_dict[ticker]['prices'])
                df.reset_index(inplace=True)
                df.set_index("date", inplace=True)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))

        except:
            pass

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    errors = 0
    
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('formatted_date', inplace=True)

            df.rename(columns={'adjclose': ticker}, inplace=True)
            df.drop(['date','index','close','high','low','open','volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        except:
            errors += 1
            pass

    print(main_df.tail())
    print('Errors: {}'.format(errors))
    main_df.to_csv('sp500_joined_closes.csv')



#save_sp500_tickers()
#get_data_from_yahoo()
compile_data()