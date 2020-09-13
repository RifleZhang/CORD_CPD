import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_EPS = 1e-10

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)
        # pre_msg [batch, time_step, edge, 2*dim]

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                pre_msg.size(2), self.msg_out_shape)
        # all_msgs [batch, time_step, edge, hidden]

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:]
            all_msgs += msg
        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        # agg_msgs [batch, time_step, atom, hidden]

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward_reconstruct(self, inputs, rel_type, rel_rec, rel_send, pred_steps=5):
        # inputs [batch, num_atom, time_step, dim]
        inputs = inputs.transpose(1, 2).contiguous()
        # inputs [batch, time_step, num_atom, dim]
        batch, time_steps, num_atom, dim = inputs.shape
        # shape rel_type: [batch, time_step, edge, relation]
        # left_time_steps = ((time_steps - 1) // pred_steps + 1) * pred_steps - time_steps
        # rels = [rel_type] + [rel_type[:, -1, :, :]] * left_time_steps
        # rel_type = torch.cat(rels, 1)

        assert (pred_steps <= time_steps)
        preds = []
        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, :-pred_steps, :, :]

        # Run n prediction steps
        for start in range(pred_steps):
            end = time_steps - pred_steps + start
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, rel_type[:, start:end])
            preds.append(last_pred)
        sizes = [batch, time_steps-pred_steps, pred_steps, num_atom, dim]
        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, :, i, :, :] = preds[i]
        # output [batch, time_step-pred_step, pred_step, num_atom, dim]

        return output.contiguous()

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        # inputs [batch, atom, time_step, dim]
        inputs = inputs.transpose(1, 2).contiguous()
        # inputs [batch, time_step, atom, dim]
        # shape rel_type: [batch, time_step, edge, relation]
        time_steps = inputs.size(1)
        left_time_steps = ((time_steps - 1) // pred_steps + 1) * pred_steps - time_steps
        rels = [rel_type] + [rel_type[:, -1, :, :]] * left_time_steps
        rel_type = torch.cat(rels, 1)

        assert (pred_steps <= time_steps)
        preds = []
        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        # curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 rel_type[:, step::pred_steps, :, :])
            preds.append(last_pred)
        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = torch.zeros(sizes)
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):
        # inputs [batch, atom, timestep, dims]
        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)
        # inputs [batch, timestep, atom, hidden]
        # rel_type [batch, timestep, edge, relation]
        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = F.gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()

class RecurrentBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""

    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0.):
        super(RecurrentBaseline, self).__init__()
        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.rnn = nn.LSTM(n_atoms * n_hid, n_atoms * n_hid, n_layers)
        self.fc2_1 = nn.Linear(n_atoms * n_hid, n_atoms * n_hid)
        self.fc2_2 = nn.Linear(n_atoms * n_hid, n_atoms * n_out)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def step(self, ins, hidden=None):
        # Input shape: [num_sims, n_atoms, n_in]
        x = F.relu(self.fc1_1(ins))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        x = x.view(ins.size(0), -1)
        # [num_sims, n_atoms*n_hid]

        x = x.unsqueeze(0)
        x, hidden = self.rnn(x, hidden)
        x = x[0, :, :]

        x = F.relu(self.fc2_1(x))
        x = self.fc2_2(x)
        # [num_sims, n_out*n_atoms]

        x = x.view(ins.size(0), ins.size(1), -1)
        # [num_sims, n_atoms, n_out]

        # Predict position/velocity difference
        x = x + ins

        return x, hidden



    def forward(self, inputs, prediction_steps, burn_in=False, burn_in_steps=1, forecast_step=1):

        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        outputs = []
        hidden = None

        for step in range(0, inputs.size(2) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]
            else:
                # Use ground truth trajectory input vs. last prediction
                if not step % prediction_steps:
                    ins = inputs[:, :, step, :]
                else:
                    ins = outputs[step - 1]

            output_forecast = []
            output, hidden = self.step(ins, hidden)
            hidden_forecast = hidden
            output_forecast.append(output)
            for i in range(forecast_step-1):
                output, hidden_forecast = self.step(output, hidden_forecast)
                output_forecast.append(output)

            output_forecast = torch.stack(output_forecast, dim=2)
            # Predict position/velocity difference
            outputs.append(output_forecast)

        outputs = torch.stack(outputs, dim=3)

        return outputs
