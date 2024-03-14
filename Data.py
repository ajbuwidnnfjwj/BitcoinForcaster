import pyupbit
from sklearn import preprocessing
import numpy as np
import torch
import datetime
from datetime import timedelta
import urllib.request
import json


#call coin value data from upbit
class coinData:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day200").reset_index()
        self.scalerX = preprocessing.MinMaxScaler()
        df[['open', 'high', 'low', 'volume']] = self.scalerX.fit_transform(
            df[['open', 'high', 'low', 'volume']])
        self.scalerY = preprocessing.MinMaxScaler()
        df[['close']] = self.scalerY.fit_transform(df[['close']])
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

    def getTrendRatio(self):
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
        request.add_header("X-Naver-Client-Id"," ")
        request.add_header("X-Naver-Client-Secret"," ")
        request.add_header("Content-Type","application/json")
        self.response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        return self.response.getcode()


class Data(coinData, trendData):
    def __init__(self, seq_length, split, batch_size, *args):
        coinData.__init__(self, seq_length)
        trendData.__init__(self, args)
        self.batch_size = batch_size
        self.split = split  #for test
        self.input_size=0
        self.x_seq = []; self.y_seq = []
        self.train_loader=None


    def getTrainset(self):
        self.getTrendRatio()
        self.trend_rat = self.trend_rat.reshape(200,1)
        self.dataX = np.c_[self.dataX, self.trend_rat]
        for i in range(len(self.dataX)-self.seq_length):
            self.x_seq.append(self.dataX[i:i+self.seq_length])
            self.y_seq.append(self.dataY[i+self.seq_length])

        self.x_seq=np.array(self.x_seq); self.y_seq=np.array(self.y_seq)
        x_seq = torch.FloatTensor(self.x_seq); self.input_size = x_seq.size(2)
        y_seq = torch.FloatTensor(self.y_seq).view([-1,1])
        train = torch.utils.data.TensorDataset(x_seq, y_seq)
        self.train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.batch_size, shuffle=False)
        return self.train_loader
    
    @property
    def PredictSet(self):
        temp = np.array([self.x_seq[-1]])
        return torch.FloatTensor(temp)
