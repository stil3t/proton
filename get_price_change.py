import moexalgo as algo
import datetime as dt
import pandas as pd

def transform_date(date):
    return dt.datetime(date.year,
                       date.month,
                       date.day,
                       date.hour,
                       date.minute - date.minute % 5)


def get_price_change(ticker, period='week'):
    if period == 'week':
        delta = dt.timedelta(days=7)
    elif period == 'year':
        delta = dt.timedelta(years=1)
    elif period == 'month':
        delta = dt.timedelta(days=30)
    elif period == 'day':
        delta = dt.timedelta(days=1)
        
        
    beg1 = transform_date(today - delta)
    beg2 = beg1 + dt.timedelta(days=2)
    end1 = transform_date(today)
    end2 = end1 - dt.timedelta(days=2)
    
    tckr = algo.Ticker(ticker)
    
    beg = tckr.tradestats(date=beg1.date(), till_date=beg2.date()).iloc[-1]
    end = tckr.tradestats(date=end2.date(), till_date=end1.date()).iloc[0]
    return (end.pr_vwap - beg.pr_vwap) / beg.pr_vwap * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker')
    parser.add_argument('--period', default='month')
    args = parser.parse_args()
    prch = get_price_change(args.ticker, args.period)
    print(prch)