from shared_layers import CustomActivationFunction
import torch
from torch import nn

# Linear Embedding
class LinearEmbedding(nn.Module):
    def __init__(self, d_model, linear: bool = False):
        super(LinearEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(4, d_model)
        self.relu = nn.ReLU()
        self.linear = linear

    def forward(self, x: Tensor):
        if self.linear:
            return self.embedding_layer(x)
        else:
            return self.relu(self.embedding_layer(x))

# Sine positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6, base=10000.):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a 2D positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for the batch size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input Shape: (batch_size, max_seq_len, d_model)
        Output Shape: (batch_size, max_seq_len, d_model)
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# Subclassed transformer with custom activation function
class CustomTransformer(nn.Transformer):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation=None,
                 custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05,
                 batch_first=False, norm_first=False, device=None, dtype=None):
        super().__init__(d_model, nhead, num_encoder_layers,
                         num_decoder_layers, dim_feedforward, dropout,
                         activation, custom_encoder, custom_decoder, layer_norm_eps,
                         batch_first, norm_first, device, dtype)

        # Assign custom activation function
        self._activation_fn = CustomActivationFunction()

        # Overwrite activation function for transformer encoder layers and decoder layers
        for i in range(self.encoder.num_layers):
            self.encoder.layers[i].linear1.activation = self._activation_fn
            self.encoder.layers[i].linear2.activation = self._activation_fn

        for i in range(self.decoder.num_layers):
            self.decoder.layers[i].linear1.activation = self._activation_fn
            self.decoder.layers[i].linear2.activation = self._activation_fn


# Transformer autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout, device):
        super(TransformerAutoencoder, self).__init__()
        self.trans = CustomTransformer(d_model=d_model, nhead=num_heads,
                                       num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                       dim_feedforward=d_ff, dropout=dropout,
                                       activation='relu', custom_encoder=None,
                                       custom_decoder=None, layer_norm_eps=1e-05,
                                       batch_first=True, norm_first=False,
                                       device=device, dtype=None)
        self.embedding = LinearEmbedding(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, base=100)
        self.custom_act = CustomActivationFunction()
        self.dense = nn.Linear(d_model, 128)
        self.output = nn.Linear(128, 4)

    def forward(self, src, tgt):
        src_mask = (src[:,:,0] == 0)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)
        return self.output(self.custom_act(self.dense(self.trans(src, tgt, src_key_padding_mask=src_mask))))