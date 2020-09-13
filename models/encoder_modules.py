import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CorrelationEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, enc_spatial, enc_temporal, do_prob=0., factor=True):
        super(CorrelationEncoder, self).__init__()

        self.factor = factor
        self.dropout_prob = do_prob
        self.enc_temporal = enc_temporal
        self.enc_spatial = enc_spatial
        self.mlp_in = MLP(n_in, n_hid, n_hid, do_prob)
        self.nhead = 8
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

        if self.enc_spatial == "transformer":
            print("Use transformer as spatial encoder")
            self.transformerLayerSpatial = nn.TransformerEncoderLayer(n_hid, nhead=self.nhead, dim_feedforward=self.nhead*n_hid)
            self.transformerEncoderSpatial = nn.TransformerEncoder(self.transformerLayerSpatial, num_layers=1)
            self.mlp_out = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        else:
            self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            if self.factor:
                self.mlp_out = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            else:
                self.mlp_out = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        if self.enc_temporal == "transformer":
            print("Use transformer as temporal encoder")
            self.posEncoder1 = PositionalEncoding(n_hid, self.dropout_prob)
            self.posEncoder2 = PositionalEncoding(n_hid, self.dropout_prob)
            self.transformerLayer1 = nn.TransformerEncoderLayer(n_hid, nhead=self.nhead, dim_feedforward=self.nhead*n_hid, dropout=self.dropout_prob)
            self.transformerLayer2 = nn.TransformerEncoderLayer(n_hid, nhead=self.nhead, dim_feedforward=self.nhead*n_hid, dropout=self.dropout_prob)
            self.transformerEncoder1 = nn.TransformerEncoder(self.transformerLayer1, num_layers=1)
            self.transformerEncoder2 = nn.TransformerEncoder(self.transformerLayer2, num_layers=1)
        elif self.enc_temporal == "rnn":
            print("Use bidirectional rnn as temporal encoder")
            self.gru1 = torch.nn.GRU(n_hid, n_hid//2, 1, bidirectional=True)
            self.gru2 = torch.nn.GRU(n_hid, n_hid//2, 1, bidirectional=True)
        else:
            print("Use unidirectional rnn as temporal encoder")
            self.gru1 = torch.nn.GRU(n_hid, n_hid//2, 1, bidirectional=False)
            self.gru2 = torch.nn.GRU(n_hid, n_hid//2, 1, bidirectional=False)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)


    def RNN(self, gru, x, num_sims, num_timesteps, num_atoms, num_hidden):
        # shape of x [num_timesteps * num_sims, num_atoms, num_hidden]
        x = x.view([num_timesteps, num_sims * num_atoms, num_hidden])
        x, _ = gru(x)
        x = x.view([num_timesteps * num_sims, num_atoms, -1])
        x = F.dropout(x, self.dropout_prob, training=self.training)
        return x

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        # incoming doesn't include itself, n-1 edges
        return incoming / (incoming.size(1) - 1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def transform_temporal(self, transformer, posEncoding, x, num_sims, num_timesteps, num_atoms, num_hidden):
        # shape of x [num_timesteps * num_sims, num_atoms, num_hidden]
        x = x.view([num_timesteps, num_sims * num_atoms, num_hidden])
        if posEncoding is not None:
            x = posEncoding(x)
        x = transformer(x)
        x = x.view([num_timesteps * num_sims, num_atoms, -1])
        x = F.dropout(x, self.dropout_prob, training=self.training)
        return x

    def transform_spatial(self, transformer, posEncoding, x, num_sims, num_timesteps, num_atoms, num_hidden):
        # shape of x [num_timesteps * num_sims, num_atoms, num_hidden]
        x = x.transpose(0, 1)
        # shape of x [num_atoms, num_timesteps * num_sims, num_hidden]
        x = transformer(x)
        x = x.transpose(0, 1).contiguous()
        x = F.dropout(x, self.dropout_prob, training=self.training)
        return x

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        num_sims, num_atoms, num_timesteps, _ = inputs.size()
        # x = inputs.view(inputs.size(0), inputs.size(1), -1)
        x = inputs.permute(2, 0, 1, 3).contiguous().view([inputs.size(2)*inputs.size(0), inputs.size(1), -1])
        # New shape: [num_timesteps*num_sims, num_atoms, num_dims]

        x = self.mlp_in(x)  # 2-layer ELU net per node


        # temporal encoding
        if self.enc_temporal == 'transformer':
            x = self.transform_temporal(self.transformerEncoder1, self.posEncoder1, x, num_sims, num_timesteps, num_atoms, x.size(-1))
        else:
            x = self.RNN(self.gru1, x, num_sims, num_timesteps, num_atoms, x.size(-1))


        # spatial encoding
        if self.enc_spatial == 'transformer':
            x = self.transform_spatial(self.transformerEncoderSpatial, None, x, num_sims, num_timesteps, num_atoms, x.size(-1))
        else:
            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x_skip = x
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)

        # temporal encoding again
        if self.enc_temporal == 'transformer':
            x = self.transform_temporal(self.transformerEncoder2, self.posEncoder2, x, num_sims, num_timesteps, num_atoms, x.size(-1))
        else:
            x = self.RNN(self.gru2, x, num_sims, num_timesteps, num_atoms, x.size(-1))


        # output edge
        if self.enc_spatial == 'transformer':
            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp_out(x)
        else:
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp_out(x)


        rel_type = self.fc_out(x)
        rel_type = rel_type.view([num_timesteps, num_sims, num_atoms * (num_atoms-1), -1])
        rel_type = rel_type.transpose(0, 1)

        return rel_type
