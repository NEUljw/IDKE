import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from read_data import coding_sentences

# parameters
train_data_number = 10
batch_size = 5
epoch_num = 2
dev_data_number = 3
lr = 0.001
pad_size = 20
src_len = pad_size     # src sentence length(after padding)
tgt_len = pad_size     # tgt sentence length(after padding and add 'S' 'E')
d_model = 128   # Embedding Size 512
d_ff = 512     # FeedForward dimension 2048
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 2    # number of Encoder of Decoder Layer
n_heads = 8     # number of heads in Multi-Head Attention

src_vocab, tgt_vocab, src_input, tgt_input, tgt_output, dev_input, dev_endings, dev_output, dev_answers = coding_sentences(
    max_seq_len=pad_size, train_data_number=train_data_number, dev_data_number=dev_data_number)

# 模型输入输出的格式：
# S: decoder输入的起始标识符
# E: decoder输出的结束标识符
# P: padding标识符
# sentences = ['ich mochte ein P', 'S i want a beer', 'i want a beer E']
# 经过上面的函数处理后变为词典编号的格式，作为模型的输入输出

src_input = torch.tensor(src_input, dtype=torch.long)
tgt_input = torch.tensor(tgt_input, dtype=torch.long)
tgt_output = torch.tensor(tgt_output, dtype=torch.long)
dev_input = torch.tensor(dev_input, dtype=torch.long)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)


def positional_coding_cal(data, max_len):
    data = data.tolist()
    position_list = []
    for i in data:
        del_zero = []
        for j in i:
            if j != 0:
                del_zero.append(j)
        rr = [k+1 for k in range(len(del_zero))]
        if len(rr) < max_len:
            rr += [0 for _ in range(max_len-len(rr))]
        position_list.append(rr)
    return torch.tensor(position_list, dtype=torch.long)


# 位置编码公式的实现
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 维度为奇数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 维度为偶数
    return torch.tensor(sinusoid_table, dtype=torch.float32)


# padding mask
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# sequence mask
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)  # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)

        pad_row = torch.zeros([1, d_model], dtype=torch.float32)
        position_encoding = torch.cat((pad_row, get_sinusoid_encoding_table(src_len, d_model)))

        self.pos_emb = nn.Embedding(src_len+1, d_model)
        self.pos_emb.weight = nn.Parameter(position_encoding, requires_grad=False)

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        # print('enc inputs:')
        # print(enc_inputs)
        # print(self.src_emb)
        # print(self.pos_emb)
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs += self.pos_emb(positional_coding_cal(enc_inputs, max_len=src_len))
        # print(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)

        pad_row = torch.zeros([1, d_model], dtype=torch.float32)
        position_encoding = torch.cat((pad_row, get_sinusoid_encoding_table(tgt_len, d_model)))

        self.pos_emb = nn.Embedding(tgt_len + 1, d_model)
        self.pos_emb.weight = nn.Parameter(position_encoding, requires_grad=False)

        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs += self.pos_emb(positional_coding_cal(dec_inputs, max_len=tgt_len))
        # self-attention用到padding mask和sequence mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # decoder-encoder attention只用到padding mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def batch_data(data):
    batch_num = math.ceil(data.size(0)/batch_size)   # 向上取整
    batches = data.chunk(batch_num, dim=0)     # tensor拆分
    return batches


def train(model, epoch_num, lr, src_batches, tgt_batches, output_batches):    # train
    model.train()
    print('--'*20 + 'train start' + '--'*20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        print('Epoch:', '%02d' % (epoch + 1))
        epoch_loss = 0
        for batch_num in range(len(src_batches)):
            enc_inputs = src_batches[batch_num]
            dec_inputs = tgt_batches[batch_num]
            target_batch = output_batches[batch_num]

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            model.zero_grad()
            loss = criterion(outputs, target_batch.contiguous().view(-1))
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        # dev_acc = evaluate(model)
        print('epoch loss avg = ', '{:.6f}'.format(epoch_loss/len(src_batches)))
        # print('dev accuracy:', dev_acc)
        print('--'*30)
    print('--'*20 + 'train complete' + '--'*20)


def main():
    src_batches = batch_data(data=src_input)
    tgt_batches = batch_data(data=tgt_input)
    output_batches = batch_data(data=tgt_output)
    model = Transformer()
    train(model, epoch_num=epoch_num, lr=lr,
          src_batches=src_batches, tgt_batches=tgt_batches, output_batches=output_batches)


def for_eva_cal(four_endings, model_output):
    four_probs = []
    for ending_index in range(four_endings.size(0)):
        one_ending = four_endings[ending_index]
        one_prob = 0.0
        for i in range(len(one_ending)):
            # if one_ending[i].item() == 0:
            #     break
            one_prob += model_output[ending_index][i][one_ending[i].item()].item()
        four_probs.append(one_prob)
    # print(four_probs)
    return four_probs.index(max(four_probs))


def evaluate(model):    # evaluate
    test_dec_input = torch.tensor(dev_output, dtype=torch.long)    # 验证集decoder的输入
    dev_out = torch.tensor(dev_endings, dtype=torch.long)         # 与decoder输入相对应的输出
    dev_out = dev_out.split(4, 0)
    test_data = dev_input
    test_data_list = test_data.tolist()
    new_test_data = []
    for one_pre in test_data_list:
        new_test_data += [one_pre for _ in range(4)]
    new_test_data = torch.tensor(new_test_data, dtype=torch.long)  # 验证集encoder的输入
    # print(test_dec_input)
    # print(dev_out)
    # print(new_test_data)

    predict, _, _, _ = model(new_test_data, test_dec_input)
    predict = predict.view(new_test_data.size(0)//4, 4, tgt_len, len(tgt_vocab))

    predict_result = []
    for sen_index in range(len(dev_out)):
        four_ends = dev_out[sen_index]
        predict_result.append(for_eva_cal(four_ends, predict[sen_index]))

    correct_count = 0
    for i in range(len(dev_answers)):
        if predict_result[i] == dev_answers[i]:
            correct_count += 1
    return correct_count/len(dev_answers)


if __name__ == '__main__':
    main()
