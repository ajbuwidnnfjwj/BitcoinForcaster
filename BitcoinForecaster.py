import Data
import Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


if __name__ == '__main__':
    d = Data.Data(10, 0.8 ,'비트코인','코인','비트코인','업비트')
    model = Model.VanillaRNN(input_size=d.input_size,
                             hidden_size=64,
                             sequence_length=d.seq_length,
                             num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loss_list = []

    epoch = 200
    model.train()
    for e in range(epoch):
        seq, target = d.train_set, d.train_label
        out = model(seq)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {e+1}/{epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        seq = d.predict_set
        out = model(seq).squeeze(1)

    plt.plot(d.predict_label)
    plt.plot(out)
    plt.show()