# Investment program

# Import relevant libraries
import bs4 as bs
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import os
import pickle
import requests
from yahoofinancials import YahooFinancials
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# Webscrapping S&P500 tickers from wikipedia
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if '.' not in ticker:
            tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers


# Get daily data for each company 
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
    errors = []

    for ticker in tickers:
        try:
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                TKR = YahooFinancials(ticker)
                fin_dict = TKR.get_historical_price_data(start, end, time_period)
                df = pd.DataFrame(fin_dict[ticker]['prices'])
                df.reset_index(inplace=True)
                df.set_index("date", inplace=True)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            #else:
                #print('Already have {}'.format(ticker))

        except:
            print('No data available for {} ticker'.format(ticker))
            errors.append(ticker)
            pass
    print(errors)
    for e in errors:
        tickers.remove(e)
        
    print (tickers)
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)


# Compile all the adj_close data in a single csv file
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    
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
            print('Error in compile_data function')
            pass

    print(main_df.tail())
    main_df.to_csv('sp500_joined_closes.csv')


# Show a correlation heatmap between the 500 companies
def visualize_data():
    style.use('ggplot')

    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')

    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    plt.savefig("correlations.png", dpi = (300))
    plt.show()


# Process data for each label
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        try:
            df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        except:
            pass    

    df.fillna(0, inplace=True)
    return tickers, df

# Buy, Sell, Hold strategy
def buy_sell_hold(*args):
    cols = [c for c in args]
    buy_threshold = 0.028
    sell_threshold = -0.025
    for col in cols:
        if col > buy_threshold:
            return 1
        if col < sell_threshold:
            return -1
    return 0

# Define features and targets
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df

# ML function
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence

# main calls each function
if __name__ == "__main__":
    save_sp500_tickers()
    get_data_from_yahoo()
    compile_data()
    visualize_data()
    #do_ml('XOM\n')
    do_ml('AAPL\n')
    #do_ml('ABT\n')