"""
This file contains all models that could be used for the midi2params problem.

API for the forward function is forward(self, batch) where batch contains an array 'x'
of shape (N, seq_len, feats) where N=batch size, seq_len=length of each sequence in frame rate,
and feats=number of features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
from utils.util import sample_from

class NoContextLinear(nn.Module):
    """
    A no-context, vanilla linear model. Used to provide baseline.
    """
    def __init__(self, config=None):
        """
        config: config YAML file as a dictionary.
        """
        super(NoContextLinear, self).__init__()
        if config is None:
            raise 'Must provide a config to initialize a SeqModel.'

        model = config.model
        preprocessing = config.preprocessing

        input_size = model.input_size
        if model.autoregressive_type == 'onehot':
            input_size = input_size + preprocessing.cent_bins + preprocessing.ld_bins
        elif model.autoregressive_type == 'scalar':
            input_size = input_size + 2

        self.linear = nn.Linear(input_size,
                                preprocessing.cent_bins + preprocessing.ld_bins)

        self.cent_bins = preprocessing.cent_bins
        self.preprocessing = preprocessing
        self.autoregressive_type = model.autoregressive_type

    def generate(self, batch):
        """
        Do true generation. See self.forward docstring for full description of what batch contains.
        """

        preprocessing = self.preprocessing
        with torch.no_grad():
            N = batch['pitches'].shape[0]
            # convert input pitches to one-hot
            x = F.one_hot(batch['pitches'].long(), 129).float()
            x = torch.cat((x,
                   batch['onset_arr'].unsqueeze(-1),
                   batch['offset_arr'].unsqueeze(-1)), dim=-1)

            # initialize the zeroth outputs to use
            fill_val = -1 if self.autoregressive_type == 'scalar' else 0
            cent_out_i = torch.full((N, 1, preprocessing.cent_bins), fill_val).type(x.type())  # keep CUDA if CUDA
            ld_out_i = torch.full((N, 1, preprocessing.ld_bins), fill_val).type(x.type())  # keep CUDA if CUDA
            
            # initialize arrays to append outputs to
            cent_logits = []
            ld_logits = []
            cent_out = []
            ld_out = []

            # do autoregression
            for i in range(x.shape[1]):
                x_i = x[:,i].unsqueeze(1)
                if self.autoregressive_type != 'none':
                    x_i = torch.cat((x_i,
                                     cent_out_i,
                                     ld_out_i), dim=-1)

                out_i = self.linear(x_i)

                cent_logits_i, ld_logits_i = out_i[..., :preprocessing.cent_bins], out_i[..., preprocessing.cent_bins:]
                cent_logits.append(cent_logits_i)
                ld_logits.append(ld_logits_i)


                # get the output of the model now by taking a true random sample of our distribution
                # TODO: fix for self.autoregressive_type==scalar case
                cent_probs = F.softmax(cent_logits_i, dim=-1)
                cent_out_i = sample_from(cent_probs).float()
                
                ld_probs = F.softmax(ld_logits_i, dim=-1)
                ld_out_i = sample_from(ld_probs).float()
                cent_out.append(cent_out_i)
                ld_out.append(ld_out_i)


            # concatenate together cent logits and ld logits
            cent_logits = torch.cat(cent_logits, dim=1)
            ld_logits = torch.cat(ld_logits, dim=1)
            cent_out = torch.cat(cent_out, dim=1)
            ld_out = torch.cat(ld_out, dim=1)
            return cent_logits, ld_logits, cent_out, ld_out

    def forward(self, batch):
        """
        Do forward pass on the full batch.
        """

        x = batch['x']
        out = self.linear(x)
        cent_logits, ld_logits = out[..., :self.cent_bins], out[..., self.cent_bins:]
        return cent_logits, ld_logits

class NoContextDeep(nn.Module):
    """
    A no-context, deeper model.
    """
    def __init__(self):
        raise NotImplementedError
    def forward(self, batch):
        raise NotImplementedError


class ConvModel(nn.Module):
    """
    A general 1D convolutional model composed of convolutional layer(s) and
    a linear layer at the end.
    """
    def __int__(self, frame_size):
        raise NotImplementedError
    def forward(self, batch):
        raise NotImplementedError

class SeqModel(nn.Module):
    """
    A general non-attention-based sequence model composed of some kind of RNN
    followed by a linear layer applied to each hidden state in the sequence.
    """
    def __init__(self, config=None):
        """
        config: the config YAML as a dictionary
        """
        super(SeqModel, self).__init__()
        
        if config is None:
            raise 'Must provide a config to initialize a SeqModel.'

        model = config.model
        preprocessing = config.preprocessing

        if model.seq_type == 'GRU':
            seq_model = nn.GRU
        elif model.seq_type == 'RNN':
            seq_model = nn.RNN
        elif model.seq_type == 'LSTM':
            seq_model = nn.LSTM

        input_size = model.input_size
        if not(model.bidirectional):
            if model.autoregressive_type == 'onehot':
                input_size = input_size + preprocessing.cent_bins + preprocessing.ld_bins
            elif model.autoregressive_type == 'scalar':
                input_size = input_size + 2

        # NOTE: pytorch doesn't accept dropout != 0.0 if num_layers=1
        if model.num_layers == 1:
            dropout = 0.0
        else:
            dropout = model.dropout

        self.rnn = seq_model(input_size, model.hidden_size,
                             num_layers=model.num_layers, dropout=dropout,
                             bidirectional=model.bidirectional, batch_first=True)

        # setup the variable number of linear layers
        linear_in = 2 * model.hidden_size if model.bidirectional else model.hidden_size
        if model.num_linears == 0:
            self.linears = lambda x: x
        elif model.num_linears == 1:
            self.linears = nn.Linear(linear_in,
                                     preprocessing.cent_bins + preprocessing.ld_bins)
        else:
            linears = [nn.Linear(model.linear_hidden_size, model.linear_hidden_size) for i in range(model.num_linears - 2)]
            linears.append(nn.Linear(model.linear_hidden_size, preprocessing.cent_bins + preprocessing.ld_bins))
            linears.insert(0, nn.Linear(linear_in, model.linear_hidden_size))
            self.linears = nn.Sequential(*linears)

        self.autoregressive_type = model.autoregressive_type

        self.cent_bins = preprocessing.cent_bins
        self.model = model
        self.preprocessing = preprocessing
        self.config = config

    def generate(self, batch, top_k=None, top_p=None):
        """
        Do true generation. See self.forward docstring for full description of what batch contains.
        """

        preprocessing = self.preprocessing
        with torch.no_grad():
            N = batch['pitches'].shape[0]
            # convert input pitches to one-hot
            x = F.one_hot(batch['pitches'].long(), 129).float()
            x = torch.cat((x,
                   batch['onset_arr'].unsqueeze(-1),
                   batch['offset_arr'].unsqueeze(-1)), dim=-1)

            # initialize the zeroth outputs to use
            fill_val = -1 if self.autoregressive_type == 'scalar' else 0
            cent_out_i = torch.full((N, 1, preprocessing.cent_bins), fill_val).type(x.type())  # keep CUDA if CUDA
            ld_out_i = torch.full((N, 1, preprocessing.ld_bins), fill_val).type(x.type())  # keep CUDA if CUDA
            
            # initialize arrays to append outputs to
            cent_logits = []
            ld_logits = []
            cent_out = []
            ld_out = []

            # do autoregression
            for i in range(x.shape[1]):
                x_i = x[:,i].unsqueeze(1)
                if self.autoregressive_type != 'none':
                    x_i = torch.cat((x_i,
                                     cent_out_i,
                                     ld_out_i), dim=-1)

                if i == 0:
                    out_i, h_i = self.rnn(x_i)
                else:
                    out_i, h_i = self.rnn(x_i, h_i)
                out_i = self.linears(out_i)

                cent_logits_i, ld_logits_i = out_i[..., :preprocessing.cent_bins], out_i[..., preprocessing.cent_bins:]
                cent_logits.append(cent_logits_i)
                ld_logits.append(ld_logits_i)


                # get the output of the model now by taking a true random sample of our distribution
                # TODO: fix for self.autoregressive_type==scalar case
                cent_probs = F.softmax(cent_logits_i, dim=-1)
                cent_out_i = sample_from(cent_probs).float()

                ld_probs = F.softmax(ld_logits_i, dim=-1)
                ld_out_i = sample_from(ld_probs).float()
                cent_out.append(cent_out_i)
                ld_out.append(ld_out_i)


            # concatenate together cent logits and ld logits
            cent_logits = torch.cat(cent_logits, dim=1)
            ld_logits = torch.cat(ld_logits, dim=1)
            cent_out = torch.cat(cent_out, dim=1)
            ld_out = torch.cat(ld_out, dim=1)
            return cent_logits, ld_logits, cent_out, ld_out

    def forward(self, batch):
        """
        Forward-pass.
        
        batch: dictionary containing several arrays, each of which is (N, seq_len, [optional dimension]). Potentially:
            - 'f0': (N, seq_len)
            - 'loudness_db': (N, seq_len)
            - 'pitches': (N, seq_len)
            - 'onset_arr': (N, seq_len)
            - 'offset_arr': (N, seq_len)
            - 'f0_gt_discrete': (N, seq_len)
            - 'f0_gt_gauss': (N, seq_len, 101)
            - 'ld_gt_discrete': (N, seq_len)
            - 'ld_gt_gauss' (N, seq_len, 121)
        """

        if self.model.bidirectional:
            x = F.one_hot(batch['pitches'].long(), 129).float()
            x = torch.cat((x,
                           batch['onset_arr'].unsqueeze(-1),
                           batch['offset_arr'].unsqueeze(-1)), dim=-1)
        else:
            x = batch['x']

        out, _ = self.rnn(x)
        out = self.linears(out)
        cent_logits, ld_logits = out[..., :self.cent_bins], out[..., self.cent_bins:]
        return cent_logits, ld_logits

class AttentionSeqModel(nn.Module):
    """
    A general attention-based sequence model composed of some kind of RNN
    followed by a linear layer applied to each hidden state in the sequence.
    """
    def __init__(self):
        raise NotImplementedError
    def forward(self, batch):
        raise NotImplementedError

model_dict = {
    'linear-1': NoContextLinear,
    'conv-1': ConvModel,
    'seq-1': SeqModel,
    'seq-best': SeqModel
}
