import numpy as np
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import requests
import json
import re

import os
from time import sleep
import argparse

import datetime as dt

day = dt.timedelta(days=1)
two_days = dt.timedelta(days=2)
hour = dt.timedelta(hours=1)
m5 = dt.timedelta(minutes=5)


import talib
import moexalgo as algo
import natasha
import razdel

emb = natasha.NewsEmbedding()

def embed(s):
    s = s.lower()
    
    while not (s in emb) and s:
        s = s[:-1]
        
    if not s:
        return emb['<unk>']
    else:
        return emb[s]
    
def vectorize_fixed(s, target_len=20):
    s = razdel.tokenize(s)
    res = np.zeros((target_len, 300))
    for i, word in enumerate(s):
        res[i] = embed(word.text)
    
    return res

def vectorize(s):
    s = razdel.tokenize(s)
    res = np.array([emb['<pad>']])
    for i, word in enumerate(s):
        res = np.vstack([res, embed(word.text)])
    
    return res[1:]


def get_news(query, pages=1, period='month'):
    news = []
    
    for page in range(pages):
        url = f'https://mediametrics.ru/satellites/api/search/?ac=search&nolimit=0' +\
              f'&q={query}' +\
              f'&p={page}' +\
              f'&c=ru&d={period}' +\
              f'&sort=tr' +\
              f'&callback=JSONP'
        r = requests.get(url)
        news.extend(re.findall('[{][^{}]*[}]', r.text))
        
    news = pd.DataFrame(list(map(json.loads, news)))
    news.drop(['author_id', 'country', 'country_name', 'author', 'text'], axis=1, inplace=True)
    
    news.timestamp = pd.to_datetime(news.timestamp.astype('int') + 3*60*60, unit='s')
    news.traffic = news.traffic.astype('int')    
    
    return news


def get_ta_from_dataframe(dataframe, ticker, time, n_before, n_after):
    df = dataframe[data.secid == ticker].copy()
    last_dt_before = df.datetime[df.datetime <= time].max()
    first_dt_after = df.datetime[df.datetime >= time].min()
    return df[(df.datetime > last_dt_before - n_before * m5) &
              (df.datetime < first_dt_after + n_after * m5 )]

def get_ta_algopack(ticker, time, n_before, n_after):
    date0 = str((time - two_days).date())
    date1 = str((time + two_days).date())
    
    url = f'https://iss.moex.com/iss/datashop/algopack/eq/tradestats/{ticker}.csv?from={date0}&till={date1}&iss.only=data'
    df = pd.read_csv(url, sep=';', skiprows=2)
    sleep(0.005)
    
    df['datetime'] = pd.to_datetime(df.tradedate + ' ' + df.tradetime, format='%Y-%m-%d %H:%M:%S')
    df.drop(['tradedate', 'tradetime', 'SYSTIME'], axis=1, inplace=True)
    
    last_dt_before = df.datetime[df.datetime <= time].max()
    first_dt_after = df.datetime[df.datetime >= time].min()

    return df[(df.datetime > last_dt_before - n_before * m5) &
              (df.datetime < first_dt_after + n_after * m5 )]

@dataclass
class Company:
    ticker: str
    search_query: list


model = keras.models.load_model("my_model.keras")

def transform_date(date):
    return dt.datetime(date.year,
                       date.month,
                       date.day,
                       date.hour,
                       date.minute - date.minute % 5)


today = dt.datetime.today()


def get_background(ticker, search_query, period='month', alpha = .999, pages=5):
    news = get_news(search_query, pages=pages, period=period)
    news_sent = []

    for title in news.title:
        news_sent.append(model.predict(vectorize_fixed(title, 130).reshape(1, 130, 300), verbose=0).item())

    news['sent'] = news_sent

    res = pd.DataFrame(columns=['sent'], index=pd.date_range(start=transform_date(news.timestamp.min()),
                                                             end=dt.datetime.today(),
                                                             freq=m5))
    for row in news.sort_values(by='timestamp').iterrows():
        candle_time = transform_date(row[1]['timestamp'])
        sent = row[1]['sent']
        res.loc[pd.to_datetime(candle_time)] = sent

    res.fillna(0, inplace=True)
    
    for i in range(1, len(res)):
        res.iloc[i] = res.iloc[i-1] * alpha + res.iloc[i]
    
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker')
    parser.add_argument('--search_query')
    parser.add_argument('--period', default='month')
    parser.add_argument('--pages', default=5)
    args = parser.parse_args()
    sent = get_background(args.ticker, args.search_query, args.period, pages=args.pages)
    print(sent.iloc[-1].item())