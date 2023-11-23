from statsmodels.tsa.arima.model import ARIMA
from Data import Data 

class Model(Data):
    def __init__(self) -> None:
        super().__init__()
        self.model = ARIMA(self.df['close'], order = (1,len(self.diff_list)-1,1))
        self.model_fit = self.model.fit()

    def summary(self) -> None:
        return self.model_fit.summary()    

    def forecast(self) -> list:
        return self.model_fit.forecast()