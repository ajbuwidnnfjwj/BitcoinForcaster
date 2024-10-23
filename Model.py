import torch
import torch.nn as nn
import numpy as np

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size)  # 초기 hidden state 설정하기.
        out, _ = self.rnn(x, h0)  # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
        out = out.reshape(out.shape[0], -1)  # many to many 전략
        out = self.fc(out)
        return out

# Positional Encoding Layer (시계열 데이터에 필수적)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(transformer, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (sequence_length, batch_size, input_size)
        src = self.input_embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Transformer Encoder의 출력 shape: (sequence_length, batch_size, d_model)
        transformer_output = self.transformer_encoder(src)
        # 마지막 시퀀스 타임스텝의 출력만 사용
        # transformer_output의 shape은 (sequence_length, batch_size, d_model)이므로 마지막 시퀀스를 뽑아냄
        last_timestep_output = transformer_output[-1]  # (batch_size, d_model)

        # Fully Connected Layer를 통해 최종 출력 (batch_size, output_size)
        output = self.fc_out(last_timestep_output)  # output의 shape은 (batch_size, output_size)

        return output  # 이 결과는 (batch_size, 1) 형태가 됨