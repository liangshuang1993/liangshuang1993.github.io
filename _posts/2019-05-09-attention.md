### 各种attention结构，以及相应代码

在早期的机器翻译应用中，神经网络一般是如下图的Seq2seq结构，左边是encoder，右边是decoder，encoder会通过RNN将最后一个step的隐藏状态向量c作为输出，如下图所示。

![](/papers/tts/24.png)

但是这种结构存在一定的问题，encoder最后一个时刻输出的向量是定长的，会损失一定的信息，因此后面基本都采用了attention结构。

![](/papers/tts/25.png)

记encoder hidden states为$h_1,h_N \in R^{d_1}$,在第t个时间步，decoder hidden state记为$s_t\in R^{d_2}$, 可以依此计算出attention score $e^t$

$$e^t=[s^T_th_1,...,s^T_th_N]$$

之后用softmax计算出attention distribution $\alpha^t$

$$\alpha^t=softmax(e^t)$$

计算attention output $a_t$

$$a_t=\sum_{i=1}^N\alpha^t_ih_i$$
最后，将attention output $a_t$和decoder hidden state $s_t$拼起来，就可以将其试做没有attention的普通的seq2seq模型了。

$$[a_t;s_t]$$

attention其实有很多变种
首先，计算attention score也有很多方法：
- dot-product: $s^Th_i\in R$, 要求$d_1=d_2$
- multiplicative attention: $s^TWh_i \in R, W \in R^{d_2*d_1}$ 
- additive attention: $v^Ttanh(W_1h_i+W_2s)$, $v \in R^{d_3}$是weight vector, $W_1 \in R^{d_3*d_1}, W_2 \in R^{d_3*d_2}, d_3$是超参

additive attention实现起来有一些差异，有一种是BahdanauAttention， 它是使用上一时刻的隐状态$s_{i-1}$进行计算，还有一种是LuongAttention，是用当前时刻的隐状态$s_i$进行计算。

下面用代码实现(_reference:https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py_)：
__pytorch版__
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Example:
        >>> attention = Attention(256)
        >>> context = Variable(torch.randn(5, 3, 256))
        >>> output = Variable(torch.randn(5, 5, 256))
        >>> output, attn = attention(output, context)
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None
    
    def set_mask(self, mask):
        """
        sets indices to be masked

        Args: mask(torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask
    
    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2 * dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
```

顺便再学写一下seq2seq model的代码：
```python
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, input_lengths=None, target_variable=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_varible, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
```
下面是decoder(包含attention):
```python
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.function as F

from .attention import Attention

""" A base class for RNN. """
import torch.nn as nn


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')
    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.
    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DecoderRNN(BaseRNN):
    """
        Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, sos_id, 
                 eos_id, n_layers=1, rnn_cell='gru', bidirectional=False, 
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size, 
                                         input_dropout_p, dropout_p, n_layers, 
                                         rnn_cell)
        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        if use_attention:
            self.attention = Attention(self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embeded = self.embedding(input_var)
        embeded = self.input_drop(embeded)

        output, hidden = self.rnn(embeded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)
        
        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)),
                                     dim=1).view(batch_size. output_size, -1)
        return predicted_softmax, hidden, attn
    
    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)

            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        
        # Manual unrolling is used to support random teacher forcing
        # If teacher_force_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols
        
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """
        Initialize the decoder hidden state.
        """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """
        If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0: h.size(0): 2], h[1: h.size(0): 2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError('Argument encoder_outputs cannot be None when attention is used.')
        
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if self.rnn_cell is nn.LSTM:
                batch_size = encoder_hidden[0].size(1)
            elif self.rnn_cell is nn.GRU:
                batch_size = encoder_hidden.size(1)
        
        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError('Teacher forcing has to be disabled (set 0) when no inputs is provided.')
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1
        
        return inputs, batch_size, max_length
```

可以再看下TTS-mozilla中additive attention的写法
```python
import torch
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(atten_dim, 1, bias=False)

    def forward(self, annots, query):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)

        alignment = self.v(torch.tanh(processed_query + processed_annots))
        return alignments.squeeze(-1)


class AttentionRNNCell(nn.Module):
    def __init__(self, out_dim, rnn_dim, annot_dim, memory_dim, windowing=False):
        """
        General Attention RNN Wrapper

        Args:
            out_dim: context vector feature dimension
            rnn_dim: rnn hidden state dimension
            annot_dim: annotation vector feature dimension
            memory_dim: memory vector feature dimension
            windowing: attention windowing forcing monotonic attention. It is only active in eval mode.
        """
        super(AttentionRNNCell, self).__init__()
        self.rnn_cell = nn.GRUCell(annot_dim + memory_dim, rnn_dim)
        self.windowing = windowing
        if self.windowing:
            self.win_back = 3
            self.win_front = 6
            self.win_idx = None
        self.alignment_model = BahdanauAttention(annot_dim, rnn_dim, out_dim)

    def forward(self, memory, context, rnn_state, annots, attn, mask, t):
        """
        Args:
            memory: prenet output, input of decoder
            context: output[1] of this cell
            rnn_state: output[0] of this cell, hidden state of decoder RNN
            annots: encoder output
            mask: attention mask for sequence padding
            t: time index
        """
        if t == 0:
            self.alignment_model.reset()
            self.win_idx = 0
        
        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i-1}, s_{i-1})
        rnn_output = self.rnn_cell(torch.cat((memory, context), -1), rnn_state)

        # Alignment
        # (batch, max_tim)
        # e_{ij} = a(s_{i-1}, h_{j})
        alignment = self.alignment_model(annots, rnn_output)
        if mask is not None:
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float('inf'))
        
        # windowing
        if not self.training and self.windowing:
            back_win = self.win_idx - self.win_back
            front_win = self.win_idx + self.win_front
            if back_win > 0:
                alignment[:, :back_win] = -float('inf')
            if front_win < memory.shape[1]:
                alignment[:, front_win:] = -float('inf')
            
            self.win_idx = torch.argmax(alignment, 1).long()[0].item()
        
        # Normalize context weight
        alignment = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(dim=1).unsqueeze(1)

        # Attention context vector
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment
```

__tensorflow版__

首先学习下基础的RNN以及seq2seq的tensorflow实现。
tensorflow有两种RNN，动态RNN和静态RNN，貌似主流是用动态RNN。根据https://stackoverflow.com/questions/39734146/whats-the-difference-between-tensorflow-dynamic-rnn-and-rnn 可以看出两者区别：

> tf.nn.rnn创建一个固定长度的RNN没有展开的图，这就意味着如果你用200时间步的输入调用了tf.nn.rnn，你是创建了一个200RNN steps的静态图。首先图的创建非常慢，其次，你不能传入长度超过200的序列。tf.nn.dynamic_rnn解决了这个问题，它是使用了tf.while动态地创建图，这就可以使得图的创建更快，并且可以传入不定长的序列。

```python
import numpy as np
import tensorflow as tf

tf.reset_default_graph() # can use this to clean default graph and nodes.
sess = tf.InteractiveSession()

PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2 # encoder is bidirectional

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None, ), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

# encoder
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
# encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
#     encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=True
# )
# when time_major=False, inputs' shape is [batch_size, sequence_length, embedding_size], otherwise, [sequence_length, batch_size, embedding_size]
(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                cell_bw=encoder_cell,
                                inputs=encoder_inputs_embedded,
                                sequence_length=encoder_inputs_length,
                                dtype=tf.float32,
                                time_major=True)
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), axis=2)
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), axis=1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), axis=1)

encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

# decoder
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32, time_major=True, scope='plain_decoder'
)
docoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction = tf.argmax(docoder_logits, 2)
```

tensorflow中contrib.seq2seq.python.ops包含4大块：
- attention mechanism: 不同的attention机制，包括
    - _BaseAttentionMechanism: 所有Attention基类
        - BahdanauAttention
        - LuongAttention
        - 等
    - AttentionWrapperState类： 用来存储整个计算过程中的state，类似rnn的state，只不过这里还额外存储了attention, time等信息
    - AttentionWrapper类：用来组件rnn cell和上述所有类的实例，从而构建一个带有attention机制的decoder
    - 一些公用方法，包括求解attention权重的softmax的替代函数hardmax，不同attention的权重计算函数

BahdanauAttention就是additive attention。有两种实现形式：一种是原始的，没有正则化，一种是正则化的。

```python
__init__(
    num_units, # the depth of the query mechanism
    memory,    # the memory to query; usually the output of an RNN encoder. This tensor should be shapped [batch_size, max_tim, ...] 
    memory_sequence_length=None, # sequence lengths for the batch entries in memory. If provided, the memory tensor rows are masked with zeros for values past the respective sequence lengths
    normalize=False, 
    probability_fn=None, # Convert score to probabilities. the default is tf.nn.softmax. other options include tf.contrib.seq2seq.hardmax and tf.contrib.sparsemax.sparsemax
    score_mask_value=None, # the mask value for score before passing into probability_fn, only used if memory_sequence_length is not None
    dtype=None, # the data type for the query and the memory layer of the attention mechanism
    name='BahdanauAttention'
)
```

选择好attention mechanism，搭建好RNN后，需要将他们拼在一起，这里是由AttentionWrapper完成的,这个类是从RNNCell继承来的
```python
def __init__(
    cell,                      # an instance of RNNCell
    attention_mechanism,       # a list of AttentionMechanism instances or a single instance
    attention_layer_size=None, # a list of Python integers or a single Python integer, the depth of the attention(output) layer(s). 
                               # If None, use the context as attention at each time step. Otherwise feed the context and cell output into the attention layer to generate attention at each time step. 
                               # 具体而言，是在_compute_attention方法中将context和output进行concat，之后经过线性映射，变成维度为attention_layer_size的向量
                               # If attention_mechanism is a list, attention_layer_size must be a list of the same length. If attention_layer is set, this must be None
    alignment_history=False,   # boolean, whether to store alignment history from all time steps in the final output state 
                               # 主要用于后期的可视化，关注attention的关注点
    cell_input_fn=None,        # default is: lambda inputs, attention: array_ops.concat([inputs, attention], -1) 
    output_attention=True,     # boolean, if True the output at each time step is the attention value. This is the behavior of Luong-style attention mechanisms.
                               # If False the output at each time step is the output of cell. This is the behavior of Bhadanau-style attention mechanisms. 
                               # In both cases, the attention tensor is propagated to the next time step via the state and is used there. 
                               # This flag only controls whether the attention mechanism is propagated up to the next cell in an RNN stack or to the top RNN output.
    initial_cell_state=None,   # the initial state value to use for the cell when the user calls zero_state(). Note that if this value is provided now, 
                               # and the user uses a batch_size argument of zero_state which does not match the batch size of initia_cell_state, proer behavior is not guaranteed.
    name=None,
    attention_layer=None       # a list of tf.layes.Layer instances or a single tf,layers.Layer instance taking the context and cell output as inputs to generate attention at each time step. 
                               # If None, use the context as attention at each time step. If attention_mechanism is a list, attention_layer must be a list of the same length. 
                               # If attention_layers_size is set, this must be None.
)

```
一个例子
```python
def _create_rnn_cell(self):
    def single_rnn_cell():
        single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
        return cell
    cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
    return cell

decoder_cell = self._create_rnn_cell()

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=self.rnn_size)

training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embeded, sequence_length=self.decoder_targets, time_major=False)
training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=decoder_initial_state,
                                                   output_layer=output_layer)
decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, impute_finished=True,
                                                          maximum_iterations=self.max_target_sequence_length)
# decoder_outputs is [rnn_outputs, sample_id]
# rnn_outputs: [batch_size, decoder_tagets_length, vocab_size]
# sample_id: [batch_size], tf.int32 保存最终的编码结果，可以表示最后的答案
```

tensorflow的seq2seq中也提供了beamsearch
```python
if useBeamSearch > 1:
    decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=useBeamSearch)
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, tokens_go, w2i_target['EOS'], decoder_initial_state, beam_width=useBeamSearch, output_layer=tf.layers.Dense(vocab_size))
else:
    decoder_initial_state = encoder_final_state
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=tf.layers.Dense(vocab_size))

decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decoder(decoder, maximum_iterations=tf.reduce_max(tf.seq_targets_length))


```

再来学习以下tacotron1中有attention的seq2seq实现：
models文件夹中有五个文件：
```
|--__init__.py
|--helpers.py
|--modules.py
|--rnn_wrappers.py
|--tacotron.py
```

rnn_wrappers.py定义了两个wrapper

```python
class DecoderPrenetWrapper(RNNCell):
    '''Run RNN inputs through a prenet before sending them to the cell.'''
    def __init__(self, cell, is_training, layer_sizes):
        self._cell = cell
        self._is_training = is_training
        self._layer_sizes = layer_size
    
    def call(self, inputs, state):
        prenet_out = prenet(inputs, self._is_training, self._layer_sizes)
        return self._cell(prenet_out, state)
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
    
    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
```

tacotron.py里面定义了seq2seq模型结构
```python
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputPreojectionWrapper, ResidualWrapper

attention_cell = AttentionWrapper(
    GRUCell(hp.attention_depth),
    BahdanauAttention(hp.attention_depth, encoder_outputs),
    alignment_history=True,
    output_attention=False
)

attention_cell = DecoderPrenetWrapper(attention_cell, is_training, hp.prenet_depth)

# concatenate attention context vector and attention RNN cell output into a 2*attention_depth=512D vector
# to form the input to the decoder RNN.
concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)

# Decoder, layers specified bottom to top
decoder_cell = MultiRNNCell([
    OutputProjectionWrapper(concat_cell, hp.decoder_depth),
    ResidualWrapper(GRUCell(hp.decoder_depth)),
    ResidualWrapper(GRUCell(hp.decoder_depth))
], state_is_tuple=True
)

# project to r mel-spectrograms
output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.r)
decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

if is_training:
    helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.r)
else:
    helper = TacoTestHelper(batch_size, hp.num_mels, hp.r)

(decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
    BasicDecoder(output_cell, helper, decoder_init_state),
    maximum_iterations=hp.max_iters
)

```

---

Tacotron2中还用到了location sensitive attention，https://arxiv.org/pdf/1506.07503.pdf

定义输入为$x=(x_1, ..., x_{L})$,经过encoder的输出为$h=(h_1,...,h_L)$,
第i步
$$\alpha_i=Attend(s_{i-1},\alpha_{i-1}, h)$$
$$g_i=\sum^L_{j=1}\alpha_{i,j}h_j$$
$$y_i\sim Generate(s_{i-1}, g_i)$$

回忆下Bahdanau attention的公式:

$$e_{i,j}=w^Ttanh(Ws_{i-1}+Vh_j+b)$$
location-sensitive attention利用了上一时刻产生的alignment信息。对每个位置j前一时刻的alignment$\alpha_{i-1}$提取k个向量$f_{i,j}\in R^k$：
$$f_i=F*\alpha_{i-1}, F\in R^{k*r}$$

$$e_{i,j}=w^Ttanh(Ws_{i-1}+Vh_j+Uf_{i,j}+b)$$

下面看下tacotron2里面是怎么实现这种attention的

```python
def _location_sensitive_score(W_query, W_fil, W_keys):
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable('attention_variable_projection', shape=[num_units], dtype=dtype,
                          initializer=tf.xavier_initializer())
    b_a = tf.get_variable('attention_bias', shape=[num_units], dtype=dtype,
                          initializer=tf.zeros_initializer())
    return tf.reduce_mean(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


class LocationSensitiveAttention(BahdanauAttention):
    def __init__(self, num_units, memory, hparams, is_training, mask_encoder=True,
                 memory_sequence_length=None, smoothing=False, cumulcate_weights=True):
        """
        Args:
            num_units: the depth of the query mechanism
            memory: the memory to query, usually the output of an RNN encoder
            smoothing: determines which normalization function to use. default is softmax. 
                       If smoothing is enabled, use a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
                       This is mainly used if the model wants to attend to multiple input parts
        """
        normalization_function = _smoothing_normalization if smoothing else None
        memory_length = memory_sequence_length if mask_encoder else None
        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn=normalization_function
        )
        self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
                                                     kernel_size=hparams.attention_kernel,
                                                     padding='same',
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer())
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False, dtype=tf.float32)
        self._cumulate = cumulcate_weights
        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(hparams.attention_win_size, dtype=tf.int32)
        self.constraint_type = hparams.synthesis_constraint_type

    def __call__(self, query, state, prev_max_attentions):
        previous_alignments = state
        with variable_scope.variable_scope(None, 'Location_Sensitive_Attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            processed_query = tf.expand_dims(processed_query, 1)

            expand_alignments = tf.expand_dims(previous_alignments, axis=2)
            f = self.location_convolution(expand_alignments)
            processed_location_features = self.location_layer(f)
            energy = _location_sensitive_score(processed_query, processed_location, self.keys)
        
        alignments = self._probability_fn(energy, previous_alignments)
        max_attentions = tf.argmax(alignments, -1, output_typtf.int32)

        if self._cumulate:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments
        
        return alignments, next_state, max_attentions
```

---

最近还有一种叫做forward attention

https://arxiv.org/pdf/1807.06736.pdf

这篇论文中提到了一种对attention alignment的一种改进方法，__windowing__, 在每个时间步只考虑固定长度$\hat x=[x_{p-2}, ..., x_{p+w}]$，这种方法可以使得对齐更稳定，并且减少计算量。最新版本的tacotron2加入了这部分代码。

在*语音合成*领域中，x和o之间的对齐，表示了linguistic feature是如何映射到对应的acoustic feature上的。在合成数十帧的时候，attention还是应当集中在一个phone上面，之后再逐渐移动到下一个phone上面，方向是单调的。这种说法和之前看到一篇论文中说法是一致的。

在每个时间步t, 定义$q_t$是output sequence的query（通常是decoder RNN的hidden state），记$\pi \in \{1,2,...,N\}$是隐变量类别，代表根据分布$p(\pi_t|x, q_t)$所选择的hidden representation。
$y_t(n)=p(\pi_t=n|x, q_t)$, 可以得到$c_t=\sum^N_{n=1}y_t(n)x_n$。

假定$\pi_t$仅与$x$和$q_t$有关。则记alignment path为
$$p(\pi_{1:t}|x,q_{1:t})=\prod^t_{t'=1}p(\pi_{t'}|x,q_{t'})=\prod^t_{t'=1}y_{t'}(\pi_{t'})$$,初始化$\pi_0=1$

和CTC的计算类似，前向变量

![](/papers/tts/26.png)

可以通过$\alpha_{t-1}(n)$和$\alpha_{t-1}(n-1)$递归地求$\alpha_{t}(n)$

$$\alpha_{t}(n)=(\alpha_{t-1}(n) + \alpha_{t-1}(n-1)y_t(n))$$

定义
![](/papers/tts/27.png)

可以由下式计算context vector:
$$c_t=\sum^N_{n=1}\hat\alpha_t(x)x(n)$$

完整的算法流程图如下所示：

![](/papers/tts/38.png)

文章还提出了一种改进后的方法**forward attention with transition agent**：

定义一个由一层hidden layer和sigmoid output组成的transition agent DNN，每个时间步输出为$u_t \in (0,1)$，$u_t$表示第t个时间步attended phone应该向前移动的概率，这个DNN网络的输入是$c_t, o_{t-1}, q_t$拼接起来的向量。

算法流程如下：

![](/papers/tts/39.png)



---

下面介绍下大名鼎鼎的Transformer

在序列模型中,很多都用到了RNN层.还有很多网络是序列计算的,这就使得我们无法采用并行计算的方式.本文中提出了一种叫做Transformer的模型,它不再采用RNN,而是全都是用了attention,可以进行并行化.

模型结构如下图所示:

![](/papers/tts/55.png)

下面介绍下Transformer结构,并附上pytorch代码(http://nlp.seas.harvard.edu/2018/04/03/attention.html), 模型仍然是encoder-decoder结构

```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Generator(nn.Module):
    """softmax"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

**encoder**
encoder中包含6个一样的层.每层都含有两个子层,第一个是多头self-attention,第二层是position-wise全连接层.在这两个子层中,都加了residual,并加入了layer normalization.为了维度上可以直接相加,模型中的embedding层和所有的sub-layer,维度都是512维.

```python
def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameters(torch.ones(features))
        self.b_2 = nn.Parameters(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = dropout
    
    def forward(self, x, sublayer):
        # 这里的layer normalization被移到了每个sub-layer的输入. 和GPT2中的用法一样.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```


**decoder**
decoder中也是包含6个完全一样的层.除了encoder中介绍的那两种层之外,还引入了一个multi-head attention,作用在encoder的输出上.和encoder中类似,在每个子层上添加了residual connections以及layer normalization.另外还修改了decoder中的self-attention层,添加了mask,防止拿到当前位置之后的信息.


```python
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    # np.triu 返回矩阵上三角部分,其余定义为0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

**attention**

![](/papers/tts/56.png)

attention的输入有三种数据,分别为query, keys, values. query和keys的维度为$d_k$,values的维度为$d_v$.

$$Attention(Q, K, V) = softmax(\frac{QK}{\sqrt{d_k}})V$$

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

**multi-head attention**

多头attention可以让模型同时从不同的表征空间中,注意到不同位置的信息.不像普通的attention那样,直接将 queries,keys, values投影到$d_{model}$维,而是分h次,将它们投影到{d_k,d_k,d_v}上.

$$MultiHead(Q, K, V)=Concat(head_1, ..., head_h)W^O$$
$$head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)$$

即先将query, key, value分h次投射到不同的维度上,$W^Q_i \in R^{d_{model}\times d_k},W^K_i \in R^{d_{model}\times d_k},W^V_i \in R^{d_{model}\times d_v}, W^O \in R^{hd_v\times d_{model}}$

在实验中采用的是$h=8,d_k=d_v=d_{model}/h=64$


模型中不同位置attention的使用方法:
- encoder-decoder attention处,query来自decoder中的前一层,keys和values来自encoder的输出.这和普通的seq2seq模型类似.

- encoder中的self-attention层,key,value,query来自同一个地方,即encoder前一层的输出.

- decoder中的self-attention层,为了防止信息向左流动,我们将这里的attention加入了mask.


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # do all the linear projections in batch
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concat
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

除了attention层之外,encoder和decoder中的每一层都包含了一个全连接网络,包含两层线性变换,之间有ReLU,input和output的维度为512,中间层的维度是2048.

```python
class PisitionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PisitionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w_1(x))))
```


**position encoding**

还有一个很重要的内容是position encoding.由于我们的模型中没有CNN和RNN,我们还需要注入一些位置信息.如果将K,V按行打乱训练,attention得到的结果还是一样的.因此我们将input embedding加入positional encoding.positional encoding和embedding的维度相同,都是$d_{model}$.

$$PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
```

整体模型代码:

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model= EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```

由下表可以看到,当序列长度n小于维度d时,self-attention层比RNN层更快.
> attention层的好处在于可以一步到位捕捉到全局的联系,因为它直接将序列两两比较(代价是计算量变为$O(n^2)$,当然由于是纯矩阵运算,这个计算量相当也不是很严重);相比之下,RNN层需要一步步地推才能捕捉到,而CNN则需要通过层叠来扩大感受野,这是Attention明显的优势.


(引用自 https://kexue.fm/archives/4765)
![](/papers/tts/57.png)
