import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class self_Attention(nn.Module):
    def __init__(self, embed_size = 512, heads = 8):
        super(self_Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        # 개별 attention의 embed_size 설정
        self.head_dim = embed_size // heads

        self.query = nn.Linear(in_features = self.head_dim, out_features = self.head_dim, bias = False)
        self.key = nn.Linear(in_features = self.head_dim, out_features = self.head_dim, bias = False)
        self.value = nn.Linear(in_features = self.head_dim, out_features = self.head_dim, bias = False)

        # 8개의 attention을 1개의 attention으로 통합
        # 8 * 64 = 512
        self.fc_out = nn.Liear(in_features = heads * self.head_dim, out_features = embed_size)


    def forward(self, query, key, value, mask):

        # 총 문장 개수
        N_batch = query.shape[0]
        query_len = query.shape[1] # query token 개수
        key_len = key.shape[1] # key token 개수
        value_len = value.shape[1] # value token 개수

        # Batch → Heads → Sequence → Features(head_dim)
        value = value.reshape(
            N_batch, self.haeds, value_len, self.head_dim
        )
        key = key.reshape(
            N_batch, self.heads, key_len, self.head_dim
        )
        query = query.reshape(
            N_batch, self.heads, query_len, self.head_dim
        )

        V = self.value(value)
        K = self.key(key)
        Q = self.query(query)


        # Q: (N, H, L, D)
        # K: (N, H, L, D)
        # attention_score는 Q와 K의 내적이기에 (L x D) x (D X L)

        # Q:           (N, H, L, D)
        # K^T:         (N, H, D, L)
        # --------------------------------
        # Q @ K^T ==>  (N, H, L, L)   # 우리가 원하는 score shape

        attention_score = torch.matmul(Q, K.transpose(-2, -1))

        if mask is not None:
            # mask 값으로 -inf를 넘겨야 하기 때문에 이를 -1e20으로 대입하여 수행
            attention_score = attention_score.masked_fill(mask == 0, float('-1e20'))

        # Attention(Q,K,V)=softmax(Q*K^T/sqrt(d_k)V
        d_k = self.embed_size ** 0.5
        softmax_score = torch.softmax(attention_score / d_k, dim = 3)
        # softmax_score shape = (n, h, query_len, key_len)

        # softmax * Value => attention 통합을 위한 reshape

        # softmax_score: (n, h, query_len, key_len)
        # V:             (n, h, key_len,   head_dim)
        # -------------------------------------------
        # out:           (n, h, query_len, head_dim)
        out = torch.matmul(softmax_score, V).reshape(
            N_batch, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_size= 512, heads=8, dropout=0.1, forward_expansion=2):

#        embed_size(=512) : embedding 차원
#        heads(=8) : Attention 개수
#        dropout(=0.1): Node 학습 비율
#       forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
#                                forward_expension * embed_size(2*512 = 1024)

        super(EncoderBlock, self).__init__()

        self.attention = self_Attention(embed_size = embed_size, heads = heads)

        self.norm1 = nn.LayerNorm(embed_size) # 512
        self.norm2 = nn.LayerNorm(embed_size) # 512

        self.feed_forward = nn.Sequential(
            # 512 -> 1024
            nn.Linear(embed_size, forward_expansion * embed_size),
            # ReLU
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        '''
        Attention Out = MHA(Q,K,V)
        x1 = LayerNorm(Q + Attention Out)
        FFN Out = FFN(x1)
        output = layerNorm(x1 + FFN Out)
        '''
        # self Attention
        attention = self.attention(value = value, key = key, query = query, mask = mask)
        # Add & Normalization
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(Encoder, self).__init__()

        '''
        src_vocab_size(=11509) : input vocab 개수
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        heads(=8) : Attention 개수
        device : cpu;
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        dropout(=0.1): Node 학습 비율
        max_length : batch 문장 내 최대 token 개수(src_token_len)
        '''

        self.embed_size = embed_size
        self.device = device

        # input + positional_embedding
        self_word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)
        pos_embed.requires_grad = False # 논문에서 false로 수행함
        # [max_len] → unsqueeze → [max_len, 1]
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)


        self.layers = nn.ModuleList(
            [
                EncoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _, seq_len = x.size()
        pos_embed = self.pos.embed[:, :seq_len, :]

        out = self.dropout(self.word_embedding(x) + pos_embed)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()

        self.norm = nn.LayerNorm(embed_size)
        self.attention = self_Attention(embed_size, heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout(0.1)

    def forward(self, x, value, key, src_trg_mask, target_mask):

        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.encoder_bolck(value, key, query, src_trg_mask)

        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(Decoder, self).__init__()

        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        pos_embed = torch.zeros(max_length, embed_size)  # (trg_token_len, embed_size) 2
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)


        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        def forward(self, x, enc_src, src_trg_mask, trg_mask):
            _, seq_len = x.size()
            pos_embed = self.pos_embed[:, :seq_len, :]
            out = self.dropout(self.word_embedding(x) + pos_embed).to(self.device)

            for layer in self.layers:
                # Decoder Input, Encoder(K), Encoder(V) , src_trg_mask, trg_mask
                out = layer(out, enc_src, enc_src, src_trg_mask, trg_mask)

            return out
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_layers,
        forward_expansion,
        heads,
        dropout,
        device,
        max_length,
    ):

        super(Transformer, self).__init__()
        self.Encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout, max_length, device
        )

        self.Decoder =Decoder(
            trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device
        )

        self.scr_pad_idx = src_pad_idx
        self.trg_pad_idx =  trg_pad_idx
        self.device= device

    def encode(self, src):
        """
        Test 용도로 활용 encoder 기능
        """
        src_mask = self.make_pad_mask(src, src)
        return self.Encoder(src, src_mask)

    def decode(self, src, trg, enc_src):
        """
        Test 용도로 활용 decoder 기능
        """
        # decode
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_trg_mask(trg)
        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # Linear Layer
        out = self.fc_out(out)  # (n, decoder_query_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out

    def make_pad_mask(self, query, key):
        """
        Multi-head attention pad 함수
        """
        len_query, len_key = query.size(1), key.size(1)

        key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size x 1 x 1 x src_token_len) 4

        key = key.repeat(1, 1, len_query, 1)
        # (batch_size x 1 x len_query x src_token_len) 4

        query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # (batch_size x 1 x src_token_len x 1) 4

        query = query.repeat(1, 1, 1, len_key)
        # (batch_size x 1 x src_token_len x src_token_len) 4

        mask = key & query
        return mask

    def make_trg_mask(self, trg):
        """
        Masked Multi-head attention pad 함수
        """
        # trg = triangle
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        # (n,1,src_token_len,src_token_len) 4

        trg_mask = self.make_trg_mask(trg)
        # (n,1,trg_token_len,trg_token_len) 4

        src_trg_mask = self.make_pad_mask(trg, src)
        # (n,1,trg_token_len,src_token_len) 4

        enc_src = self.Encoder(src, src_mask)
        # (n, src_token_len, embed_size) 3

        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # (n, trg_token_len, embed_size) 3

        # Linear Layer
        out = self.fc_out(out)  # embed_size => trg_vocab_size
        # (n, trg_token_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out