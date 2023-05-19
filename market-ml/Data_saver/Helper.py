import pandas as pd
import numpy as np

class normalizer:

    def norm(data):
        date_time = pd.DataFrame(data=data, columns=['date', 'open', 'high', 'low', 'close']).drop(columns=['open', 'high', 'low', 'close'])
        df = pd.DataFrame(data=data, columns=['date', 'open', 'high', 'low', 'close']).drop(columns=['date']).astype(float)

        i=0
        while i < len(df['open']):

            if df['high'][i] > df['open'][i] + df['open'].std():
                df['high'][i] = df['open'][i]

            if df['low'][i] < df['close'][i] - df['close'].std():
                df['low'][i] = df['close'][i]

            i += 1
        
        df = pd.concat(objs=[date_time, df], axis=1, join='outer')
        return df

    def normalize__with__gain(df, k):
        
        date_time = df.drop(columns=['open', 'high', 'low', 'close'])
        df = df.drop(columns=['date']).astype(float)
        
        # gain = [(df['high'].max()/df['open'][0])-1, (df['low'].min()/df['open'][0])-1]
        gain = [[np.subtract(np.divide(df['high'].max(), df['open'][k]), 1), df['high'].idxmax() - k], [np.subtract(np.divide(df['low'].min(), df['open'][k]), 1), df['low'].idxmin() - k]]
        # print(gain)

        date = np.divide(np.subtract(df, df.min()), np.subtract(df.max(), df.min()))

        df = pd.concat(objs=[date_time, date], axis=1, join='outer')

        return df, gain