import Data
import Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Model import transformer

if __name__ == '__main__':
    d = Data.Data(10, 0.8 ,'비트코인','코인','비트코인','업비트')

    sequence_length = 10
    num_features = 3  # low, high, volume, open
    data_size = 200  # 200일간의 데이터
    output_size = 1  # 예측해야 할 값 (close)

    # 모델 초기화
    d_model = 64
    nhead = 4
    num_layers = 3
    model = Model.transformer(input_size=num_features, d_model=d_model,
                                      nhead=nhead, num_layers=num_layers, output_size=output_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epoch = 200
    for e in range(epoch):
        model.train()

        seq, target = d.train_set, d.train_label
        out = model(seq.transpose(0,1))
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print(f"Epoch {e}/{epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        seq = d.predict_set
        out = model(seq.transpose(0,1))
        tensor_for_plot = [0 for _ in range(sequence_length)]
        for i in out:
            for j in i:
                tensor_for_plot.append(j)

    plt.plot(d.dataY[int(d.split*len(d.dataY)):])
    plt.plot(tensor_for_plot)
    plt.show()