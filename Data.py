import pyupbit
from sklearn import preprocessing
import numpy as np
import torch
import datetime
from datetime import timedelta
import urllib.request
import json

import naver_api_id


#call coin value data from upbit
class coinData:
    def __init__(self):
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day200").reset_index()
        self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        df[['open', 'high', 'low', 'volume']] = self.scaler.fit_transform(
            df[['open', 'high', 'low', 'volume']])
        df[['close']] = self.scaler.fit_transform(df[['close']])
        self.dataX = df[['volume', 'close']].values
        self.dataY = df['close'].values


#call NAVER trend ratio
class trendData:
    def __init__(self, keyword_arg):
        self.__keyword_arg = list(keyword_arg)
        interval = timedelta(days=199)
        today = datetime.date.today()
        self.body = {
            'startDate':(today-interval).strftime('%Y-%m-%d'),
            'endDate':today.strftime('%Y-%m-%d'),
            'timeUnit':'date',
            'keywordGroups':None
        }
        self.respose = None
        self.trend_rat = []
        self.__getTrendRatio()

    def __getTrendRatio(self):
        rescode = self.__makeRequest()
        if rescode == 200:
            json_obj = json.loads(self.response.read().decode('utf-8'))
            for ratio in json_obj['results'][0]['data']:
                self.trend_rat.append(ratio['ratio'])

            for i in range(0,200-len(self.trend_rat)):
                self.trend_rat.append(self.trend_rat[-1])
            self.trend_rat = np.array([self.trend_rat])
        else:
            print("Naver API Request Fail. Exit with Error Code "+rescode)

    def __makeBody(self):
        keyword = {}
        keyword["groupName"] = self.__keyword_arg[0]
        keyword["keywords"] = self.__keyword_arg[1:]
        self.body['keywordGroups'] = [keyword]
        self.body = json.dumps(self.body)
        return self.body

    def __makeRequest(self) -> int:
        body = self.__makeBody()
        request = urllib.request.Request("https://openapi.naver.com/v1/datalab/search")
        request.add_header("X-Naver-Client-Id",f"{naver_api_id.X_CLIENT_ID}")
        request.add_header("X-Naver-Client-Secret",f"{naver_api_id.X_CLIENT_SECRETE}")
        request.add_header("Content-Type","application/json")
        self.response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        return self.response.getcode()


class Data(coinData, trendData):
    def __init__(self, seq_length, *args):
        coinData.__init__(self)
        trendData.__init__(self, args)

        self.trend_rat = self.trend_rat.reshape(200,1)
        self.dataX = np.c_[self.dataX, self.trend_rat]
        self.dataX = torch.tensor([
            self.dataX[i:i+seq_length] for i in range(len(self.dataX)-seq_length)
        ])
        self.input_size = self.dataX.size(2)

    @property
    def train_set(self, split):
        assert 0 < split <= 1, 'variable split out of bound'
        split_len = int(len(self.dataX) * split)
        return self.dataX[:split_len]

    @property
    def predict_set(self, split):
        assert 0 < split <= 1, 'variable split out of bound'
        split_len = int(len(self.dataX) * split)
        return self.dataX[split_len:]