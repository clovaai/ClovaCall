import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Location-based
    """
    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.conv_dim = conv_dim
        self.attn_dim = attn_dim
        self.smoothing= smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        param:quries: Decoder hidden states, Shape=(B,1,dec_D)
        param:values: Encoder outputs, Shape=(B,enc_T,enc_D)
        param:last_attn: Attention weight of previous step, Shape=(batch, enc_T)
        """
        batch_size = queries.size(0)
        dec_feat_dim = queries.size(2)
        enc_feat_len = values.size(1)

        # conv_attn = (B, enc_T, conv_D)
        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2)

        # (B, enc_T)
        score =  self.fc(self.tanh(
         self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1)


        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(score) 

        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D) 
        context = torch.bmm(attn_weight.unsqueeze(dim=1), values)

        return context, attn_weight