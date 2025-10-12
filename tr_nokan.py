import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math 
import torch.nn.functional as F

import pdb; pdb.set_trace()

# Reference code - https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
# all d_ are dimensions, d_model must be divisible by num_heads.
class multiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(multiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def attention(self, Q, K, V, mask=None):
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill( mask == 0, -1e9)

        attention_probability = torch.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_probability, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.attention(Q,K,V, mask)

        out = self.W_o(self.combine_heads(attention_output))
        return out

# ----------------------------------------------------------------------------------------------------
# Base activation is har-coded to SiLU
# https://github.com/1ssb/torchkan/blob/main/KALnet.py
# efficient_kan also used SiLU activation.
# polynomial order is starting with linear, splines (efficient_kan), Legendre polynomial (see link)
# MLP is linear-Relu(linear) KAN is f2(f1(x))...
# ----------------------------------------------------------------------------------------------------

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=2,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            stand_alone_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1,1],):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) /grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,-1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size+spline_order))
        if stand_alone_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.stand_alone_spline = stand_alone_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # set some random data
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1/2)* self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.stand_alone_spline else 1.0)*
                                          self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order],
                                                           noise,))
            if self.stand_alone_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a= math.sqrt(5) * self.scale_spline)
    #------------------------------------------------------------------------------------------------------------------------
    # 2d splines
    # These splines do not scale for large models.
    # some have tried Rational Base functions (F(x) = P(x)/Q(x) = {a_0 + a_1x+...a_mx(^m)}/{1 + |b_1x+b_2x^2+...b_nx^n|}
    # I am only going to try exponential function y = e^(ax), This will help me in creating exponential maps on manifolds.
    #  -- here a is the parameter learnt during training. which is really Sigma_0^n (2^n).x^n/n! for a=2.
    # I did not add any term at start i.e. f(x) = be^(ax), where b and a are parameters as then at x=0, function = 1, property
    # vanishes.
    #------------------------------------------------------------------------------------------------------------------------
    # code tailored from efficient-kan
    def b_splines(self, x: torch.Tensor):
        # generates a grid like tensor pytorch bs.
        grid: torch.Tensor = (self.grid)
        x = x.unsqueeze(-1)
        bases = ((x >=grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order +1):
            bases = (
                        (x - grid[:, : -(k+1)]) 
                        / (grid[:, k:-1] - grid[:, : -(k+1)]) 
                        * bases[:, :, :-1]
                    ) + (
                        (grid[:, k+1 :] -x) 
                        / (grid[:, k+1 :] - grid[:, 1:(-k)]) 
                        * bases[:, :, 1:]
                    )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        # compute co-efficients
        A = self.b_splines(x).transpose(0,1)
        B = y.transpose(0,1)
        solution = torch.linalg.lstsq(A,B).solution
        result = solution.permute(2,0,1) # move out_feature to first, in features and then grid_size+spline
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
                                     if self.stand_alone_spline
                                     else 1.0) # ternary syntax A if condition else B, meaning don't scale for stand alone



    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), 
                                 self.scaled_spline_weight.view(self.out_features, -1),
                                 )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class KAN(nn.Module):
    def __init__(self, layers_hidden):
        super(KAN, self).__init__()
        self.grid_size = 5
        #self.fc1 = nn.Linear(d_model, d_ff)
        #self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = torch.nn.SiLU
        self.spline_order = 3  # cubic spline are numerically most stable 
        grid_eps = 0.02
        grid_range=[-1,1]

        # we should initialize weights of MLP using kaiming, only for standalone spline.
        # we just need to track edges and train their weights.
        # d_model is input_feature, d_ff is output. X can be multidimensional tensor.
        self.layers = torch.nn.ModuleList()
        for inf, outf in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                    KANLinear(
                        inf,
                        outf,
                        grid_size=5,
                        spline_order=2,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        # kaiming for standalone spline.
                    )
             )

    def forward(self, x: torch.Tensor, update_grid=False):
        print(x.shape, x.dim())
        # We no longer have MLP madeup of linear layers + Relu
        for layers in self.layers:
            if update_grid:
                layers.update_grid(x)
            x = layers(x) # fc2(fc1(x))... functional composition.
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.shape, x.dim())
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = multiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        #breakpoint()
        #self.feed_forward = KAN([d_model, d_ff, d_model])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x,x,x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        out = self.norm2(x+self.dropout(ff_output))
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = multiHeadAttention(d_model, num_heads)
        self.cross_attention = multiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        #breakpoint()
        #self.feed_forward = KAN([d_model, d_ff, d_model])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x,x,x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x+self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        out = self.norm3(x + self.dropout(ff_output))
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask, = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        out = self.fc(dec_output)
        return out


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads=8
num_layers = 6
# feedforward layer's dimension
d_ff = 2048
max_seq_length = 100
dropout=0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

# Training the model
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr = 0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    out = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(out.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Transformer Eval
transformer.eval()

val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

with torch.no_grad():
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
