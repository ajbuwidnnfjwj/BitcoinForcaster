import Data
import Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pyupbit

def plotLoss(model, criterion=nn.MSELoss()) -> None:
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    loss_graph = []
    for e in range(model.epoch):
        running_loss = 0.0

        for datas in d.getTrainset()[0]:
            seq, target = datas
            out = model(seq)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_graph.append(running_loss / len(d.getTrainset()[0])) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,

    if model.epoch % 100 == 0:
        print('[epoch: %d] loss: %.4f'%(model.epoch, running_loss/len(d.getTrainset()[0])))
        plt.figure(figsize=(20,10))
        plt.plot(loss_graph)
        plt.show()


def plotting(d) -> int:
    with torch.no_grad():
        train_pred = []
        test_pred = []

        for data in d.loaderset[0]:
            seq, target = data
            out = model(seq)
            train_pred += out.numpy().tolist()

        for data in d.loaderset[1]:
            seq, target = data
            out = model(seq)
            test_pred += out.numpy().tolist()

    total = train_pred + test_pred
    rescaled_total = d.scalerY.inverse_transform(np.array(total).reshape(-1,1))
    rescaled_actual = d.scalerY.inverse_transform(d.dataY.reshape(-1,1))
    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
    plt.plot(rescaled_actual, '--')
    plt.plot(rescaled_total, 'b', linewidth=0.6)

    plt.legend(['train boundary', 'actual', 'prediction'])
    plt.show()
    return test_pred[-1]


if __name__ == '__main__':
    d = Data.Data(5, 140, 20)
    input_size = d.x_train_seq.size(2)
    model = Model.VanillaRNN(input_size=input_size,
                             hidden_size=8,
                             sequence_length=d.seq_length,
                             num_layers=2)
    plotLoss(model)
    df = pyupbit.get_ohlcv("KRW-BTC", interval="day200").reset_index()
    print(plotting(d))
