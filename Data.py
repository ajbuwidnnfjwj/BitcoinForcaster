import pyupbit
from sklearn import preprocessing
import numpy as np
import torch
import datetime
from datetime import timedelta
import urllib.request
import json

class coinData:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day200").reset_index()
        self.scalerX = preprocessing.MinMaxScaler()
        df[['open', 'high', 'low', 'volume']] = self.scalerX.fit_transform(
            df[['open', 'high', 'low', 'volume']])
        self.scalerY = preprocessing.MinMaxScaler()
        df[['close']] = self.scalerY.fit_transform(df[['close']])
        self.dataX = df[['open', 'high', 'low', 'volume']].values
        self.dataY = df['close'].values

    def build_dataset(self):
        x_seq = []
        y_seq = []
        for i in range(len(self.dataX) - self.seq_length):
            x_seq.append(self.dataX[i: i + self.seq_length])
            y_seq.append(self.dataY[i + self.seq_length])
        return torch.FloatTensor(x_seq), torch.FloatTensor(y_seq).view([-1, 1])


class trendData:
    def __init__(self, *keyword_arg):
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

    def getTrendRatio(self):
        rescode = self.__makeRequest()
        if rescode == 200:
            json_obj = json.loads(self.response.read().decode('utf-8'))
            trend_rat = []
            for ratio in json_obj['results'][0]['data']:
                trend_rat.append(ratio['ratio'])

            for i in range(0,200-len(trend_rat)):
                trend_rat.append(trend_rat[-1])
            trend_rat = np.array([trend_rat])
            return trend_rat
        else:
            print("Naver API Request Fail. Exit with Error Code "+rescode)
            return None

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
        request.add_header("X-Naver-Client-Id","SdSA3fxVZrSyqUrZoD81")
        request.add_header("X-Naver-Client-Secret","0ipG8dAvMV")
        request.add_header("Content-Type","application/json")
        self.response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        return self.response.getcode()


class Data(coinData):
    def __init__(self, seq_length, split, batch_size):
        super().__init__(seq_length)
        self.batch_size = batch_size
        x_seq, y_seq = self.build_dataset()
        self.x_train_seq = x_seq[:split]
        self.y_train_seq = y_seq[:split]
        self.x_test_seq = x_seq[split:]
        self.y_test_seq = y_seq[split:]
        self.loaderset = []

    def getTrainset(self):
        train = torch.utils.data.TensorDataset(self.x_train_seq, self.y_train_seq)
        test = torch.utils.data.TensorDataset(self.x_test_seq, self.y_test_seq)
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False)
        self.loaderset = [train_loader, test_loader]
        return self.loaderset