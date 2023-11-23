import matplotlib.pyplot as plt
import pyupbit
import pandas as pd

class Data():
    def __init__(self) -> None:
        self.df = pyupbit.get_ohlcv("KRW-BTC", interval="day200").reset_index()
        self.diff_list = [self.df['close'].diff().dropna()]
        for i in range(3):
            self.diff_list.append(self.diff_list[i].diff().dropna())
        
    def getClose(self) -> pd.DataFrame:
        return self.df['close']

    def getVolume(self) -> pd.DataFrame:
        return self.df['volume']
        
class DataVisualiser(Data):
    def __init__(self) -> None:
        super().__init__()    
    
    def show_price(self) -> None:
        plt.plot(self.df['index'], self.df['close'])
        plt.show()
    
    def show_volume(self) -> None:
        plt.plot(self.df['index'], self.df['volume'])

    def show_diff(self) -> None:
        for d in range(self.diff_list):
            plt.plot(d)
            plt.show()