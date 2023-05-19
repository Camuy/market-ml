from binance import Client
from binance.client import Client
import datetime
import pandas as pd
import numpy as np
from Data_saver import Helper

def normalize(price):
    data = price.astype(float)
    data['date'] = data['date'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000))
    data = Helper.normalizer.norm(data=data)
    return data

class constant:
    api_key = '<api_key>'
    api_secret = '<api_sicret>'
    symbol = "BTCUSDT"
    client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
    columns=['date', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore']

class price:
    def day_1():
        df = normalize(pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_1DAY, start_str="400 day ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore']))
        return df
    
    def hour_12():
        df = pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_12HOUR, start_str="200 day ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore'])
        return normalize(df)
    
    def hour_4():
        df = pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_4HOUR, start_str="100 days ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore'])
        return normalize(df)
    
    def hour_1():
        df = pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_1HOUR, start_str="20 days ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore'])
        return normalize(df)
    
    def minute_30():
        df = pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_30MINUTE, start_str="10 days ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore'])
        return normalize(df)
    
    def minute_15():
        df = pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_15MINUTE, start_str="3 days ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore'])
        return normalize(df)
    
    def minute_5():
        df = normalize(pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_5MINUTE, start_str="1 days ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore']))
        return df
         
    def minute_1():
        df = normalize(pd.DataFrame(constant.client.get_historical_klines(symbol=constant.symbol, interval=constant.client.KLINE_INTERVAL_1MINUTE, start_str="600 minute ago"), columns=constant.columns).drop(columns=['volume', 'close_time', 'asset_volume', 'N_trade', 'taker_buy_base_asset', 'taker_buy_quote_asset', 'ignore']))
        return df