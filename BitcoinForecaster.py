import Data
import Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def plotLoss(model, criterion=nn.MSELoss()) -> None:
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    loss_graph = []
    for e in range(model.epoch):
        running_loss = 0.0

        for datas in d.train_loader:
            seq, target = datas
            out = model(seq)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_graph.append(running_loss / len(d.train_loader)) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,

    # if model.epoch % 100 == 0:
    #     print('[epoch: %d] loss: %.4f'%(model.epoch, running_loss/len(d.train_loader)))
    #     plt.figure(figsize=(20,10))
    #     plt.plot(loss_graph)
    #     plt.show()


def plotting(d) -> int:
    with torch.no_grad():
        pred = [[0] for _ in range(d.seq_length)]

        for data in d.train_loader:
            seq, target = data
            out = model(seq)
            pred += out.numpy().tolist()
    rescaled_pred = d.scalerY.inverse_transform(np.array(pred).reshape(-1,1))
    rescaled_actual = d.scalerY.inverse_transform(d.dataY.reshape(-1,1))
    # plt.figure(figsize=(20,10))
    #plt.plot(np.ones(100)*len(pred), np.linspace(0,1,100), '--', linewidth=0.6)
    # plt.plot(rescaled_actual, 'r--')
    # plt.plot(rescaled_pred, 'b', linewidth=0.6)

    # plt.legend(['actual', 'prediction'])
    # plt.show()


if __name__ == '__main__':
    d = Data.Data(5, 140, 20 ,'비트코인','코인','비트코인','업비트')
    d.getTrainset()
    input_size = d.input_size
    model = Model.VanillaRNN(input_size=input_size,
                             hidden_size=8,
                             sequence_length=d.seq_length,
                             num_layers=2)
    plotLoss(model)
    plotting(d)
    d.getPredictSet()
