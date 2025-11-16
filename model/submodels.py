import torch
import torchtext
import torch.distributions
from torch import nn
import torch.nn.functional as F


class AuxiliaryNet(torch.nn.Module):
    """
    Arguments
    ---------
    batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
    aux_hidden_size : Size of the hidden_state of the LSTM   (* Later BiLSTM, check dims for BiLSTM *)
    embedding_length : Embeddding dimension of GloVe word embeddings
    --------
    """

    def __init__(self, auxiliary_hidden_size, embedding_length, biDirectional=False, num_layers=1, tau=1):
        super(AuxiliaryNet, self).__init__()
        self.hidden_size = auxiliary_hidden_size
        self.embedding_length = embedding_length
        self.biDirectional = biDirectional
        self.num_layers = num_layers

        self.aux_lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=self.biDirectional,
                                num_layers=self.num_layers, batch_first=True)  # Dropout
        if (self.biDirectional):
            self.aux_linear = nn.Linear(self.hidden_size * 2, 1)
        else:
            self.aux_linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.tau = tau

    def forward(self, input_sequence, is_train=True):

        # input : Dimensions (batch_size x seq_len x embedding_length)

        # LSTM编码
        out_lstm, (final_hidden_state, final_cell_state) = self.aux_lstm(
            input_sequence)  # ouput dim: ( batch_size x seq_len x hidden_size )

        # # 如果是双向LSTM，使用最后时刻的拼接隐藏状态；否则使用最后时刻的隐藏状态
        # if self.biDirectional:
        #     # 双向LSTM: 连接前向和后向的最后隐藏状态
        #     final_hidden = torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim=1)
        # else:
        #     # 单向LSTM: 使用最后层的最后隐藏状态
        #     final_hidden = final_hidden_state[-1,:]

        # 如果是双向LSTM，使用最后时刻的拼接隐藏状态；否则使用最后时刻的隐藏状态
        if self.biDirectional:
            # 双向LSTM: 连接前向和后向的最后隐藏状态
            # Handle both 2D and 3D cases
            if final_hidden_state.dim() == 3:
                final_hidden = torch.cat((final_hidden_state[-2, :, :], final_hidden_state[-1, :, :]), dim=1)
            else:
                final_hidden = torch.cat((final_hidden_state[-2, :], final_hidden_state[-1, :]), dim=0)
        else:
            # 单向LSTM: 使用最后层的最后隐藏状态
            # Handle both 2D and 3D cases
            if final_hidden_state.dim() == 3:
                final_hidden = final_hidden_state[-1, :, :]
            else:
                final_hidden = final_hidden_state[-1, :]

        # 得到每个时间步的概率𝑝𝑡，范围在[0,1]
        out_linear = self.aux_linear(final_hidden)  # p_t dim: ( batch_size x seq_len x 1)
        p_t = self.sigmoid(out_linear)

        if is_train:
            p_t = p_t.repeat(1, 2) # 扩展为两个类别
            p_t[:, 0] = 1 - p_t[:, 0] # 第一个类别是“不选”
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)
            g_t = g_hat[:, 1:2] # 取第二个类别”选中 “

        else:
            # size : same as p_t [ batch_size x seq_len x 1]
            m = torch.distributions.bernoulli.Bernoulli(p_t)
            g_t = m.sample()

        return g_t

        # return p_t


class BackboneNet(torch.nn.Module):
    """
      Arguments
      ---------
      batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
      backbone_hidden_size : Size of the hidden_state of the LSTM   (* Later BiLSTM, check dims for BiLSTM *)
      embedding_length : Embeddding dimension of GloVe word embeddings
      --------
      """

    def __init__(self, batch_size, backbone_hidden_size, embedding_length, biDirectional=False, num_layers=2):
        super(BackboneNet, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = backbone_hidden_size
        self.embedding_length = embedding_length
        self.biDirectional = biDirectional
        self.num_layers = num_layers

        self.backbone_lstm = nn.LSTM(self.embedding_length, self.hidden_size, bidirectional=self.biDirectional,
                                     batch_first=True, num_layers=self.num_layers)  # Dropout

    def forward(self, input_sequence, batch_size=None):
        out_lstm, (final_hidden_state, final_cell_state) = self.backbone_lstm(
            input_sequence)  # ouput dim: ( batch_size x seq_len x hidden_size )
        return out_lstm


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.ff_1 = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.ff_2 = nn.Linear(self.output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_1 = self.ff_1(x)
        out_relu = self.relu(out_1)
        out_2 = self.ff_2(out_relu)
        out_sigmoid = self.sigmoid(out_2)

        return out_sigmoid