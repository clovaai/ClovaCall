import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, encoder_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', 
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0, use_attention=True):
        super(DecoderRNN, self).__init__()
        
        self.output_size = vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.hidden_size + self.encoder_output_size, self.hidden_size, self.n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)
        
        self.attention = Attention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size, conv_dim=1, attn_dim=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size + self.encoder_output_size, self.output_size)


    def forward_step(self, input_var, hidden, encoder_outputs, context, attn_w, function):
        batch_size = input_var.size(0)
        dec_len = input_var.size(1)
        enc_len = encoder_outputs.size(1)
        enc_dim = encoder_outputs.size(2)
        embedded = self.embedding(input_var) # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        embedded = self.input_dropout(embedded)

        y_all = []
        attn_w_all = []
        for i in range(embedded.size(1)):
            embedded_inputs = embedded[:, i, :] # (B, dec_D)
            
            rnn_input = torch.cat([embedded_inputs, context], dim=1) # (B, dec_D + enc_D)
            rnn_input = rnn_input.unsqueeze(1) 
            output, hidden = self.rnn(rnn_input, hidden) # (B, 1, dec_D)

            context, attn_w = self.attention(output, encoder_outputs, attn_w) # (B, 1, enc_D), (B, enc_T)
            attn_w_all.append(attn_w)
            
            context = context.squeeze(1)
            output = output.squeeze(1) # (B, 1, dec_D) -> (B, dec_D)
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1) # (B, dec_D + enc_D)

            pred = function(self.fc(output), dim=-1)
            y_all.append(pred)

        if embedded.size(1) != 1:
            y_all = torch.stack(y_all, dim=1) # (B, dec_T, out_D)
            attn_w_all = torch.stack(attn_w_all, dim=1) # (B, dec_T, enc_T)
        else:
            y_all = y_all[0].unsqueeze(1) # (B, 1, out_D)
            attn_w_all = attn_w_all[0] # (B, 1, enc_T)
        
        return y_all, hidden, context, attn_w_all


    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_hidden: Encoder last hidden states, Default : None
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if teacher_forcing_ratio != 0:
            inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                function, teacher_forcing_ratio)
        else:
            batch_size = encoder_outputs.size(0)
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2)) # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1)) # (B, T)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input, 
                                                                                decoder_hidden, 
                                                                                encoder_outputs,
                                                                                context,    
                                                                                attn_w, 
                                                                                function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input, 
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols

        return decoder_outputs


    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)

        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
