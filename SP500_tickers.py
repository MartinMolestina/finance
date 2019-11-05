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

save_sp500_tickers()
get_data_from_yahoo()