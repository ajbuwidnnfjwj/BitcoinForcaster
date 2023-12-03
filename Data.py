import pyupbit
from sklearn import preprocessing
import torch


class xData:
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


class Data(xData):
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
