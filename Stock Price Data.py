import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from yahoofinancials import YahooFinancials
import pprint

style.use('ggplot')

start = str(dt.datetime(2010, 1, 1).date())
end = str(dt.datetime.now().date())
ticker = 'TSLA'
time_period = 'daily'
TSLA = YahooFinancials(ticker)

fin_dict = TSLA.get_historical_price_data(start, end, time_period)
df = pd.DataFrame(fin_dict[ticker]['prices'])

df.set_index('formatted_date', inplace=True)
df.drop(columns=['date'], inplace=True)

print(df.tail())
df.to_csv('TSLA.csv')
