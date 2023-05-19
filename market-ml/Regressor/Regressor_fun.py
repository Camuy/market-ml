import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf

from Data_saver import Helper
from Data_saver import Binance


class curve_fit():
    
    class functions():
        def linear(x, a, b, c):
            return a*(x+b)+c
        
        def quadratic(x, a, b, c, d):
            return a*(x+b)**2 + c*(x+b) + d
        
        def exponential( x, a, b, c):
            return a*np.exp(x+b)+c
        
        def logistic(x, a, b, c, d):
            return a/(1 + np.exp(b*(x+c))) + d

    def lin(self, data):
        xdata = np.linspace(0, 1, np.prod(data.shape))
        popt, pcov = cf(self.functions.linear, xdata, data)
        y_fit = self.functions.linear(xdata, *popt)
        err = [np.subtract(1, np.absolute(((np.sum(np.multiply(data-y_fit, xdata))/np.sum(np.multiply(data, xdata)))))),
                np.subtract(1, np.absolute(((np.sum(np.multiply(np.power(data-y_fit, 2), xdata))/np.sum(np.multiply(np.power(data, 2), xdata))))))]
        # var_exp = 1 - abs(sum((Price-yfit_exp).*(length(time)-line))./sum(Price.*(length(time)-line)))
        # r_sqr = 1 - sum((length(time)-line).*(Price-yfit_exp).^2)./sum((length(time)-line).*Price.^2)
        fun = {
            'name': 'linear',
            'function': self.functions.linear,
            'parameter': popt,
            'error': err, 
            'time': len(xdata)
            }
        return fun
    
    def quad(self, data):
        xdata = np.linspace(0, 1, np.prod(data.shape))
        popt, pcov = cf(self.functions.quadratic, xdata, data)
        y_fit = self.functions.quadratic(xdata, *popt)
        err = [np.subtract(1, np.absolute(((np.sum(np.multiply(data-y_fit, xdata))/np.sum(np.multiply(data, xdata)))))),
                np.subtract(1, np.absolute(((np.sum(np.multiply(np.power(data-y_fit, 2), xdata))/np.sum(np.multiply(np.power(data, 2), xdata))))))]
        fun = {
            'name': 'quadratic',
            'function': self.functions.quadratic,
            'parameter': popt,
            'error': err,
            'time': len(xdata)
            }
        return fun
                 
    def exp(self, data):
        xdata = np.linspace(0, 1, np.prod(data.shape))
        popt, pcov = cf(self.functions.exponential, xdata, data)
        y_fit = self.functions.exponential(xdata, *popt)
        err = [np.subtract(1, np.absolute(((np.sum(np.multiply(data-y_fit, xdata))/np.sum(np.multiply(data, xdata)))))),
                np.subtract(1, np.absolute(((np.sum(np.multiply(np.power(data-y_fit, 2), xdata))/np.sum(np.multiply(np.power(data, 2), xdata))))))]
        fun = {
            'name': 'exponential',
            'function': self.functions.exponential,
            'parameter': popt,
            'error': err,
            'time': len(xdata)
            }
        return fun
                   
    def logis(self, data):
        xdata = np.linspace(0, 1, np.prod(data.shape))
        popt, pcov = cf(self.functions.logistic, xdata, data)
        y_fit = self.functions.logistic(xdata, *popt)
        err = [np.subtract(1, np.absolute(((np.sum(np.multiply(data-y_fit, xdata))/np.sum(np.multiply(data, xdata)))))),
                np.subtract(1, np.absolute(((np.sum(np.multiply(np.power(data-y_fit, 2), xdata))/np.sum(np.multiply(np.power(data, 2), xdata))))))]                    
        fun = {
            'name': 'logistic',
            'function': self.functions.logistic,
            'parameter': popt,
            'error': err,
            'time': len(xdata)
            # 'price': np.array(data.min(), data.max())
            }
        return fun

    def function_selection(self, data):
        r = 0
        try:
            lin = self.lin(data)
            if r < lin['error'][1]:
                r = lin['error'][1]
                regression = lin
        except:
            print('error in linear:\n')
        try:
            quad = self.quad(data)
            if r < quad['error'][1]:
                r = quad['error'][1]
                regression = quad
        except:
            print('error in quadratic:\n')
        try:
            exp = self.exp(data)
            if r < exp['error'][1]:
                r = exp['error'][1]
                regression = exp
        except:
            print('error in exponential:\n')
        try:
            logis = self.logis(data)
            if r < logis['error'][1]:
                r = logis['error'][1]
                regression = logis
        except:
            print('error in logistic:\n')
        return regression

def historical_tokenization(data):
    j = 100
    k = 0
    i = 0
    token = pd.DataFrame()
    reg = []
    while j < len(data['open']):

        r = 1
        r_sel = np.inf

        # data, gain = normalize__with__gain(data)

        while j - k > 10:
            price = pd.DataFrame(data[k:j])
            # x = norm(data=token)
            # print(token, '\n\n')
            _, gain = Helper.normalizer.normalize__with__gain(data[k:j], k)

            date_time = price.drop(columns=['open', 'high', 'low', 'close'])
            df = price.drop(columns=['date']).astype(float)
            date= np.divide(np.subtract(price, price.min()), np.subtract(price.max(), price.min()))
            df = pd.DataFrame(data=date, columns=['open', 'high', 'low', 'close'])
            data_norm = pd.concat(objs=[date_time, df], axis=1, join='outer')
            data_used = (data_norm['open']+data_norm['close'])/2

            try:
                regressionn = curve_fit().function_selection(data=data_used)
                # print(regressionn)
                r = regressionn['error'][1]
            except:
                print('error: end of FIRST tokanization')
            
            if r < r_sel:

                r_sel = r
                
                try:
                    regression = curve_fit().function_selection(data=data_used)

                    # print(regression)

                    reg = {
                            'token': i,
                            'name': regression['name'],
                            'function': regression['function'],
                            'parameter': [np.array(regression['parameter'])],
                            'error': [regression['error']],
                            'time': len(data_used)/100,
                            'gain': [np.array(gain)]
                        }
                    # print(reg)
                    data_saved = data_used # Serve per plottare i dati
                    
                except:
                    print('error: end of SECOND tokanization')
                
                
        
                regpd = pd.DataFrame.from_dict(data=reg)
                #print(regpd)
                
                # Data_tokens = pd.DataFrame(data = token)
                
                j_saved = j
                #    var = regression['error'][0]
                # r = regression['error'][1]

            j = j - 1
        
        def plot():
            time = np.linspace(0, 1, len(data_used))
            plt.plot(time, data_used, 'r*')
            print(*reg['parameter'])
            fun = reg['function'](time, *reg['parameter'])
            plt.plot(time, fun, 'b-')
            plt.show()
        
        # plot()

        token = pd.concat(objs=[token, regpd], axis=0, join='outer')
        # print(token)
        
        i = i + 1

        print('\n\n\n---------------------------\n',
                f"Regression Number: {j}",
                '\n---------------------------\n\n\n')
        
        # time = np.linspace(0, 1, len(data_saved))
        # plt.plot(time, data_saved, 'r*')
        # fun = regression['function'](time, *regression['parameter'])
        # plt.plot(time, fun, 'b-')
        # plt.show()

        k = j_saved - 1
        j = k + 100
    
    j = j - 100
    # token.pop(0)
    token.index = token['token']
    token.drop('token', inplace=True,axis=1)
    # print(token)
    token.drop('function', inplace=True,axis=1)

    token_ford = token.rename(columns={'name': 'name_test', 'parameter': 'parameter_test', 'error': 'error_test', 'time': 'time_test', 'gain': 'gain_test'})  # gain shiftato di uno dati preparatiì
    #token_ford['time_test'] = token_ford['time_test'].apply(lambda d: d)
    token_ford = token_ford.shift(-1)
    token = pd.concat([token, token_ford], axis=1)

    # token.drop('function', inplace=True,axis=1)
    # seleziona l'ultima riga e salvala in una variabile separata
    Live_Data = [pd.DataFrame(data[k:len(data['open'])]), token.iloc[-1]] # create an array of last data and Second_live_token

    token = token.dropna()

    return token, Live_Data

def live_tokenization(self, data):

    tokens, second_live = self.historical_tokenization(data=data)
    x = second_live

    return tokens, x
# !!! Finire di scrivere la selezione delle funzioni!!!

def save_results(data):
    nome_file = 'BTC_test_2_minute_5.csv'
    data.to_csv(nome_file, columns=['name', 'parameter', 'error', 'time', 'gain', 'name_test', 'parameter_test', 'error_test', 'time_test', 'gain_test'], index='token')
    
    print('\n\n',f'Dati salvati in: {nome_file}', '\n\n')

def main():
    print('Financial_Bot.Regressor started')
    data = Binance.price.minute_5()
    # print(data)
    token, live = historical_tokenization(data=data)
    save_results(token)
    print("Data have been seved\n", live)
    
    # data = Binance.price.minute_5()
    # token, live = historical_tokenization(data=data)
    # print(token, live)
    
    print('Financial_Bot.Regressor finished')


if __name__ == "__main__":
    main()