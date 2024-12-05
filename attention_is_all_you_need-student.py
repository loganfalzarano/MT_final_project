#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import math

#define a dataset class for the translation dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import tqdm
from tqdm import tqdm

import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logging.info("Using device %s", device)

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

######################################################################

def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    
    return pairs

def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
            
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    
    # #convert to LongTensors for nn.embedding
    # input_tensor = input_tensor.type(torch.LongTensor)
    # target_tensor = target_tensor.type(torch.LongTensor)
    
    return input_tensor, target_tensor

######################################################################

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, tensors_from_pair):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tensors_from_pair = tensors_from_pair

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Get original tensors
        input_tensor, target_tensor = self.tensors_from_pair(self.src_vocab, self.tgt_vocab, pair)
        
        input_tensor = input_tensor.squeeze()
        target_tensor = target_tensor.squeeze()
        
        # Encoder input: original sequence + EOS
        encoder_input = torch.cat([input_tensor, torch.tensor([EOS_index])])
        
        # Decoder input: original sequence with SOS at the beginning
        decoder_input = torch.cat([torch.tensor([SOS_index]), target_tensor])
        
        # Labels: target sequence without SOS (shifted left)
        labels = torch.cat([target_tensor, torch.tensor([EOS_index])])
        
        return encoder_input, decoder_input, labels
    
def create_causal_mask(seq_len):
    """
    Creates a triangular causal mask for decoder self-attention.
    
    Args:
        seq_len: Length of the sequence
        
    Returns:
        mask: Boolean tensor of shape (seq_len, seq_len)
              True values allow attention, False values prevent it
    """
    # Create triangular mask
    # Example for seq_len=4:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # Invert to get proper masking

def collate_fn(batch):
    # Separate the batch into individual components
    encoder_inputs, decoder_inputs, labels = zip(*batch)
    
    # Pad sequences
    encoder_inputs_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_index)
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_index)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=PAD_index)
    
    # Create masks for padded positions
    # TODO: consider using different mask for encoder and cross
    encoder_mask = (encoder_inputs_padded != PAD_index).unsqueeze(1).unsqueeze(2) 
    
    # Create decoder mask for mask self-attention
    causal_mask = create_causal_mask(decoder_inputs_padded.size(1))
    decoder_mask = (decoder_inputs_padded != PAD_index).unsqueeze(1).unsqueeze(2) & causal_mask
    
    #send all items to device
    encoder_inputs_padded = encoder_inputs_padded.to(device)
    decoder_inputs_padded = decoder_inputs_padded.to(device)
    labels_padded = labels_padded.to(device)
    encoder_mask = encoder_mask.to(device)
    decoder_mask = decoder_mask.to(device)
    
    return {
        'encoder_inputs': encoder_inputs_padded,
        'decoder_inputs': decoder_inputs_padded,
        'labels': labels_padded,
        'encoder_mask': encoder_mask, # (batch_size, 1, 1, seq_len) used for both encoder attention and cross-attention
        'decoder_mask': decoder_mask # (batch_size, 1, seq_len, seq_len)
    }

def create_dataloader(train_pairs, src_vocab, tgt_vocab, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the translation dataset with padding
    
    Args:
        train_pairs: List of (source, target) sentence pairs
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
    
    Returns:
        DataLoader instance
    """
    dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, tensors_from_pair)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

######################################################################
    
class InputEmbedding(nn.Module):
    """
    Converts input tokens to continuous vector representations using learned embeddings.
    The embeddings are scaled by sqrt(embedding_size) as described in the 
    "Attention is All You Need" paper to prevent dot products from growing too large.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_size (int): Dimension of the embedding vectors
    """
    def __init__(self, vocab_size, embedding_size):
        super(InputEmbedding, self).__init__()
        #TODO: create and embedding layer

    def forward(self, input_tensor):
        """
        Convert input token indices to embeddings and apply scaling.

        Args:
            input_tensor (torch.Tensor): Input tensor of token indices
                Shape: (batch_size, sequence_length)
                Type: Long/Int tensor containing vocabulary indices

        Returns:
            torch.Tensor: Embedded and scaled representation of the input
                Shape: (batch_size, sequence_length, embedding_size)
                Type: Float tensor containing continuous vector representations
        """
        #TODO: run input tensor through embedding
        
        #TODO: multiply by sqrt of embedding size
        
        return output
    
class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings using sinusoidal functions.
    This allows the model to take sequence order into account since attention itself
    is position-agnostic. Implementation follows the formula from "Attention is All You Need".
    
    Args:
        d_model (int): Dimension of the model's embeddings (same as embedding_size)
        seq_len (int, optional): Maximum sequence length to pre-compute positions for. Defaults to 50
        dropout (float, optional): Dropout probability. Defaults to 0.1
    """
    def __init__(self, d_model, seq_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (max_seq_length)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Compute the positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0) # (1, max_seq_length, d_model)
        # NOTE this will cause the positional encoding to be saved as part of the model's state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encodings to the input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of embeddings
                Shape: (batch_size, sequence_length, d_model)
                Type: Float tensor containing embedded representations
        
        Returns:
            torch.Tensor: Input embeddings with positional encoding added
                Shape: Same as input (batch_size, sequence_length, d_model)
                Type: Float tensor containing embedded representations with positional information
        """
        # Add positional encoding to the input
        # pe[:, :x.size(1)] slices the PE matrix to match the input sequence length
        positional_encoding = self.pe[:, :x.size(1), :].requires_grad_(False)        
        x = x + positional_encoding
        return self.dropout(x)
        
def self_attention(query, key, value, mask=None):
    """
    Computes scaled dot-product attention as described in "Attention is All You Need".
    For each position in the sequence, calculates attention weights across all other 
    positions and uses these to compute a weighted sum of values.
    
    The attention formula is: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    
    Args:
        query (torch.Tensor): Query tensors 
            Shape: (batch_size, num_heads, seq_len_q, head_dim)
        key (torch.Tensor): Key tensors
            Shape: (batch_size, num_heads, seq_len_k, head_dim)  
        value (torch.Tensor): Value tensors
            Shape: (batch_size, num_heads, seq_len_v, head_dim)
            Note: seq_len_k == seq_len_v
        mask (torch.Tensor, optional): Attention mask tensor
            Shape: (batch_size, 1, seq_len_q, seq_len_k) 
            Values: Binary tensor where 0 indicates positions to mask out
    
    Returns:
        tuple:
            - context (torch.Tensor): Attention-weighted sum of values
                Shape: (batch_size, num_heads, seq_len_q, head_dim)
            - attention_scores (torch.Tensor): Attention probabilities
                Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    
    Intermediate shapes:
        - scores: (batch_size, num_heads, seq_len_q, seq_len_k)
            Raw attention logits before softmax
        - attention_scores: Same as scores but after softmax
            These sum to 1 along the seq_len_k dimension
    """
    #TODO: compute the scaled dot product attention scores using query and key
    
    #TODO: apply the mask if it is not None to set values to be very small
    
    #TODO: compute the attention scores by applying the softmax function
    
    #TODO: apply the attention scores to the value tensor to get the context
    
    return context, attention_scores

class MultiHeadedAttention(nn.Module):
    """
    Implements multi-head attention as described in "Attention is All You Need".
    Splits the embedding dimension into multiple heads, allowing the model to jointly attend
    to information from different representation subspaces at different positions.
    Args:
        embedding_size (int): Size of the embeddings (d_model in the paper)
        n_heads (int): Number of attention heads
            Note: embedding_size must be divisible by n_heads
    
    Attributes:
        head_dim (int): Dimension of each attention head (d_k in the paper)
        W_queries (nn.Linear): Linear transformation for queries
            Shape: (embedding_size, embedding_size)
        W_keys (nn.Linear): Linear transformation for keys
            Shape: (embedding_size, embedding_size)
        W_values (nn.Linear): Linear transformation for values
            Shape: (embedding_size, embedding_size)
        W_out (nn.Linear): Linear transformation for output
            Shape: (embedding_size, embedding_size)
    """
    def __init__(self, embedding_size: int, n_heads: int):
        
        assert embedding_size % n_heads == 0, "The embedding size must be divisible by the number of heads"
        
        super(MultiHeadedAttention, self).__init__()
        
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.head_dim = embedding_size // n_heads #this is d_k in the paper
        
        #TODO: initilize 3 weight matrices for queries, keys, and values
        
        #TODO: initialize a weight matrix for the output which is applied about concatenating the heads
        
    def forward(self, query, key, value, mask=None):
        """
        Applies multi-head attention to the input queries, keys, and values.
        
        Args:
            query (torch.Tensor): Query tensor
                Shape: (batch_size, seq_len_q, embedding_size)
            key (torch.Tensor): Key tensor
                Shape: (batch_size, seq_len_k, embedding_size)
            value (torch.Tensor): Value tensor
                Shape: (batch_size, seq_len_v, embedding_size)
                Note: seq_len_k == seq_len_v
                
            NOTE in the reference implementation seq_len_q == seq_len_k == seq_len_v == 15
            since all tensors are padded to a max length of 15
                
            mask (torch.Tensor, optional): Attention mask
                Shape: (batch_size, 1, seq_len_q, seq_len_k)
                
            NOTE can also be (batch_size, 1, 1, seq_len) for encoder attention, this will cause broadcasting
            over the third dimension
        
        Returns:
            tuple:
                - output (torch.Tensor): Transformed output 
                    Shape: (batch_size, seq_len_q, embedding_size)
                - attention_scores (torch.Tensor): Attention weights for each head
                    Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        
        Intermediate shapes:
            - Q, K, V after linear transformation: 
                (batch_size, seq_len, embedding_size)
            - Q, K, V after splitting heads:
                (batch_size, n_heads, seq_len, head_dim)
            - x after self_attention:
                (batch_size, n_heads, seq_len, head_dim)
            - x after concatenating heads:
                (batch_size, seq_len, embedding_size)
        """
        
        #TODO: apply the weight matrices to the query, key, and value tensors
        
        #TODO split the queries, keys, and values into n_heads
        #HINT: you can use .permute or .view to shape the tensors
        
        #NOTE (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, n_heads, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        
        #NOTE (batch_size, n_heads, seq_len, head_dim) indicates that each head sees the whole
        # sentence but only a portion of the embedding
        
        #TODO compute the attention scores for each head
        #NOTE should be of shape (batch_size, n_heads, seq_len, seq_len) since there is an 
        # attention vector over each word in the sequence for each head
        
        #concatenate the heads
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, self.embedding_size)
        
        #TODO apply the output weight matrix
    
        return output, attention_scores
        
class LayerNorm(nn.Module):
    """
    Implements Layer Normalization as described in "Layer Normalization" (Ba et al., 2016).
    Normalizes the last dimension of the input tensor to have zero mean and unit variance,
    then applies a learnable scale (gamma) and shift (beta).
    
    Args:
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-6
    
    Attributes:
        gamma (nn.Parameter): Learnable scale parameter
            Shape: (1,)
        beta (nn.Parameter): Learnable shift parameter
            Shape: (1,)
    """
    def __init__(self, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        
        #TODO: initialize the gamma and beta parameters, should be 1 to start and 0 to start
    
        """
        Applies layer normalization to the input tensor along the last dimension.
        
        Args:
            x (torch.Tensor): Input tensor to normalize
                Shape: (..., features)
                Note: Can handle any number of leading dimensions
            NOTE for this model it will be (batch_size, seq_len, embedding_size)
        
        Returns:
            torch.Tensor: Normalized tensor with same shape as input
                Shape: (..., features)
            NOTE for this model it will be (batch_size, seq_len, embedding_size)
        
        Intermediate shapes:
            - mean: (..., 1)
                Mean computed along last dimension
            - std: (..., 1)
                Standard deviation computed along last dimension
        """
    def forward(self, x):
        #TODO: compute the mean, make sure to keep the dimension

        #TODO: compute the standard deviation, make sure to keep the dimension

        #TODO: normalize the input tensor

        return output

class FeedForward(nn.Module):
    """
    Implements the feed-forward network component of the Transformer as described in 
    "Attention is All You Need". Consists of two linear transformations with a ReLU
    activation in between.
    
    Architecture: Linear -> ReLU -> Linear
    
    Args:
        embedding_size (int): Size of input and output embeddings (d_model in paper)
        hidden_size (int): Size of hidden layer (d_ff in paper), typically 4x embedding_size
    
    Attributes:
        W_1 (nn.Linear): First linear transformation
            Shape: (embedding_size, hidden_size)
        W_2 (nn.Linear): Second linear transformation
            Shape: (hidden_size, embedding_size)
    """
    def __init__(self, embedding_size, hidden_size):
        super(FeedForward, self).__init__()
        #TODO: initialize the weight matrices, embedding_size -> hidden_size -> embedding_size
        
    def forward(self, x):
        """
        Applies feed-forward transformation to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor
                Shape: (batch_size, seq_len, embedding_size)
        
        Returns:
            torch.Tensor: Transformed tensor
                Shape: (batch_size, seq_len, embedding_size)
        
        Intermediate shapes:
            - After W_1: (batch_size, seq_len, hidden_size)
            - After ReLU: (batch_size, seq_len, hidden_size)
            - After W_2: (batch_size, seq_len, embedding_size)
        """
        #TODO: apply the feed-forward transformation, use ReLU activation
        return output

class EncoderLayer(nn.Module):
    """
    Implements a single encoder layer of the Transformer as described in "Attention is All You Need".
    Each encoder layer consists of a multi-head self-attention mechanism followed by a feed-forward
    network, with layer normalization and residual connections around each sub-layer.
    
    Architecture:
        x -> Self-Attention -> Dropout -> Add & Norm -> Feed-Forward -> Dropout -> Add & Norm
    
    Args:
        embedding_size (int): Size of input and output embeddings (d_model in paper)
        n_heads (int): Number of attention heads
        hidden_size (int): Size of feed-forward network's hidden layer
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
    """
    def __init__(self, embedding_size, n_heads, hidden_size, dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        
        #TODO: initialize MultiHeadedAttention, LayerNorm, Dropout,
        
        #TODO initialize FeedForward, LayerNorm, Dropout
        
    def forward(self, x, mask):
        """
        Processes input through one encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
                Shape: (batch_size, seq_len, embedding_size)
            mask (torch.Tensor): Attention mask for padding
                Shape: (batch_size, 1, 1, seq_len)
        
        Returns:
            tuple:
                - output (torch.Tensor): Transformed output
                    Shape: (batch_size, seq_len, embedding_size)
                - attention_scores (torch.Tensor): Self-attention weights
                    Shape: (batch_size, n_heads, seq_len, seq_len)
        
        Intermediate shapes:
            - After self-attention: (batch_size, seq_len, embedding_size)
            - After dropout_1: (batch_size, seq_len, embedding_size)
            - After norm_1: (batch_size, seq_len, embedding_size)
            - After feed-forward: (batch_size, seq_len, embedding_size)
            - After dropout_2: (batch_size, seq_len, embedding_size)
            - After norm_2: (batch_size, seq_len, embedding_size)
        """
        #TODO apply self-attention with x as query, key, and value
        
        #TODO apply dropout and residual connection, then layer normalization
        
        #TODO apply feed-forward network, dropout, residual connection, and layer normalization
        
        return x, attention_scores
    
class Encoder(nn.Module):
    """
    Implements the complete encoder stack of the Transformer as described in 
    "Attention is All You Need". Consists of N identical layers stacked sequentially.
    
    Note: This implementation assumes input embeddings and positional encodings 
    are added before being passed to the encoder.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_size (int): Size of embeddings (d_model in paper)
        n_heads (int): Number of attention heads in each layer
        hidden_size (int): Size of feed-forward network in each layer
        n_layers (int): Number of encoder layers to stack
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
    
    Attributes:
        layers (nn.ModuleList): List of EncoderLayer modules
    """
    def __init__(self, vocab_size, embedding_size, n_heads, hidden_size, n_layers, dropout_p=0.1):
        super(Encoder, self).__init__()
        
        #TODO: initialize the encoder layers using nn.ModuleList
        
    def forward(self, x, mask):
        """
        Processes input through the entire encoder stack.
        
        Args:
            x (torch.Tensor): Input tensor with embeddings
                Shape: (batch_size, seq_len, embedding_size)
            mask (torch.Tensor): Attention mask for padding
                Shape: (batch_size, 1, 1, seq_len)
            NOTE could be Shape (batch_size, 1, seq_len, seq_len) if a different mask is used for the encoder
        
        Returns:
            tuple:
                - output (torch.Tensor): Final encoder output
                    Shape: (batch_size, seq_len, embedding_size)
                - all_attention_scores (list): List of attention scores from each layer
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, seq_len, seq_len)
        
        Note:
            The output maintains the same shape as the input through all encoder layers.
            Attention scores are collected from each layer for visualization/analysis.
        """
        
        all_attention_scores = []
        
        #TODO: pass the input through each layer and collect the attention scores
        
        return x, all_attention_scores
    
class DecoderLayer(nn.Module):
    """
    Implements a single decoder layer of the Transformer as described in "Attention is All You Need".
    Each decoder layer consists of three sub-layers:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention over encoder output
    3. Feed-forward network
    Each sub-layer has residual connections and layer normalization.
    
    Architecture:
        x -> Masked Self-Attention -> Dropout -> Add & Norm 
          -> Cross-Attention -> Dropout -> Add & Norm 
          -> Feed-Forward -> Dropout -> Add & Norm
    
    Args:
        embedding_size (int): Size of input and output embeddings (d_model in paper)
        n_heads (int): Number of attention heads
        hidden_size (int): Size of feed-forward network's hidden layer
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
    
    Attributes:
        masked_mult_head_attention (MultiHeadedAttention): For self-attention with future masking
        norm_1 (LayerNorm): Layer normalization after masked attention
        multi_head_attention (MultiHeadedAttention): For cross-attention with encoder outputs
        norm_2 (LayerNorm): Layer normalization after cross-attention
        feed_forward (FeedForward): Feed-forward neural network
        norm_3 (LayerNorm): Layer normalization after feed-forward
        dropout_1/2/3 (nn.Dropout): Dropout after each sub-layer
    """
    def __init__(self, embedding_size, n_heads, hidden_size, dropout_p=0.1):
        super(DecoderLayer, self).__init__()
        
        #TODO: initialize the masked multi-head attention, LayerNorm, and Dropout
        
        #TODO: initialize the cross-attention multi-head attention, LayerNorm, and Dropout
        
        #TODO: initialize the feed-forward network, LayerNorm, and Dropout
        
    def forward(self, x, encoder_outputs, src_mask, tgt_mask):
        """
        Processes input through one decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
                Shape: (batch_size, tgt_seq_len, embedding_size)
            encoder_outputs (torch.Tensor): Output from encoder
                Shape: (batch_size, src_seq_len, embedding_size)
            src_mask (torch.Tensor): Mask for encoder outputs (padding mask)
                Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Combined causal and padding mask for decoder
                Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            tuple:
                - output (torch.Tensor): Transformed output
                    Shape: (batch_size, tgt_seq_len, embedding_size)
                - masked_attention_scores (torch.Tensor): Self-attention weights
                    Shape: (batch_size, n_heads, tgt_seq_len, tgt_seq_len)
                - cross_attention_scores (torch.Tensor): Cross-attention weights
                    Shape: (batch_size, n_heads, tgt_seq_len, src_seq_len)
        
        Intermediate shapes:
            All intermediate tensors maintain shape: (batch_size, tgt_seq_len, embedding_size)
        """
        #TODO: apply the masked multi-head attention, dropout, residual connection, and layer normalization
        
        #TODO: apply the cross-attention multi-head attention, dropout, residual connection, and layer normalization
        #NOTE think about what the query, key, and value tensors are
        
        #TODO: apply the feed-forward network, dropout, residual connection, and layer normalization
        
        return x, masked_attention_scores, cross_attention_scores
    
class Decoder(nn.Module):
    """
    Implements the complete decoder stack of the Transformer as described in
    "Attention is All You Need". Consists of N identical layers stacked sequentially.
    
    The decoder includes:
    1. Input embeddings and positional encodings
    2. Multiple decoder layers with masked self-attention, cross-attention, and feed-forward networks
    
    Args:
        vocab_size (int): Size of target vocabulary
        embedding_size (int): Size of embeddings (d_model in paper)
        n_heads (int): Number of attention heads in each layer
        hidden_size (int): Size of feed-forward network in each layer
        n_layers (int): Number of decoder layers to stack
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
    
    Attributes:
        embedding (InputEmbedding): Converts target tokens to embeddings
        positional_encoding (PositionalEncoding): Adds positional information
        layers (nn.ModuleList): List of DecoderLayer modules
    """
    def __init__(self, vocab_size, embedding_size, n_heads, hidden_size, n_layers, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        #TODO: initiliaze the layers using nn.ModuleList
        
    def forward(self, x, encoder_outputs, src_mask, tgt_mask):
        """
        Processes input through the entire decoder stack.
        
        Args:
            x (torch.Tensor): Target sequence tensor
                Shape: (batch_size, tgt_seq_len)
            encoder_outputs (torch.Tensor): Output from encoder
                Shape: (batch_size, src_seq_len, embedding_size)
            src_mask (torch.Tensor): Mask for encoder outputs (padding mask)
                Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Combined causal and padding mask for decoder
                Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            tuple:
                - output (torch.Tensor): Final decoder output
                    Shape: (batch_size, tgt_seq_len, embedding_size)
                - all_masked_attention_scores (list): Self-attention weights from each layer
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, tgt_seq_len, tgt_seq_len)
                - all_cross_attention_scores (list): Cross-attention weights from each layer
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, tgt_seq_len, src_seq_len)
        
        Note:
            Unlike the encoder, the decoder input sequence should be shifted right
            and include start/end tokens. The target mask should prevent attending
            to future positions during self-attention.
        """
        #NOTE the input to the decoder is the target sequence and the encoder outputs where the target sequence is already embedded positionally encoded
        
        all_masked_attention_scores = []
        all_cross_attention_scores = []
        
        #TODO: pass the input through each layer and collect the attention scores
            
        return x, all_masked_attention_scores, all_cross_attention_scores
    
class Transformer(nn.Module):
    """
    Implements the complete Transformer model as described in "Attention is All You Need".
    Consists of an encoder stack and a decoder stack, with input/output embeddings,
    positional encodings, and final output projection.
    
    Architecture:
        Encoder: Input Embeddings -> Positional Encoding -> N x EncoderLayer
        Decoder: Target Embeddings -> Positional Encoding -> N x DecoderLayer
        Output: Linear Projection -> LogSoftmax
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        embedding_size (int): Size of embeddings (d_model in paper)
        n_heads (int): Number of attention heads in each layer
        hidden_size (int): Size of feed-forward networks
        n_layers (int): Number of encoder/decoder layers
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, n_heads, hidden_size, n_layers, dropout_p=0.1):
        super(Transformer, self).__init__()
        
        #TODO: initialize the input embedding, positional encoding, encoder
        
        #TODO: initialize the target embedding, positional encoding, decoder
        
        #TODO: initialize the output projection layer
        
    def encode(self, src_input, src_mask):
        """
        Encodes the source sequence through embeddings, positional encoding, and encoder stack.
        
        Args:
            src_input (torch.Tensor): Source token indices
                Shape: (batch_size, src_seq_len)
            src_mask (torch.Tensor): Source padding mask
                Shape: (batch_size, 1, 1, src_seq_len)
        
        Returns:
            tuple:
                - encoder_outputs (torch.Tensor): Encoded representations
                    Shape: (batch_size, src_seq_len, embedding_size)
                - all_encoder_attention_scores (list): Attention weights from each layer
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, src_seq_len, src_seq_len)
        """
        #TODO: pass the source input through the input embedding and positional encoding and then through the encoder
        
        return encoder_outputs, all_encoder_attention_scores
    
    def decode(self, tgt_input, encoder_outputs, src_mask, tgt_mask):
        """
        Decodes the target sequence using encoder outputs and target inputs.
        
        Args:
            tgt_input (torch.Tensor): Target token indices
                Shape: (batch_size, tgt_seq_len)
            encoder_outputs (torch.Tensor): Encoded source sequence
                Shape: (batch_size, src_seq_len, embedding_size)
            src_mask (torch.Tensor): Source padding mask
                Shape: (batch_size, 1, 1, src_seq_len)
            tgt_mask (torch.Tensor): Target causal mask
                Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            tuple:
                - decoder_outputs (torch.Tensor): Decoded representations
                    Shape: (batch_size, tgt_seq_len, embedding_size)
                - all_masked_attention_scores (list): Self-attention weights
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, tgt_seq_len, tgt_seq_len)
                - all_cross_attention_scores (list): Cross-attention weights
                    Length: n_layers
                    Each element shape: (batch_size, n_heads, tgt_seq_len, src_seq_len)
        """
        #TODO: pass the target input through the target embedding and positional encoding and then through the decoder
        
        return decoder_outputs, all_masked_attention_scores, all_cross_attention_scores
    
    def get_predictions(self, decoder_outputs):
        return F.log_softmax(self.projection_linear_layer(decoder_outputs), dim=-1)

######################################################################

def train_batch(batch, transformer, optimizer, criterion, max_length=MAX_LENGTH, attention_weights_through_time=None):
    encoder_inputs = batch['encoder_inputs']
    decoder_inputs = batch['decoder_inputs']
    labels = batch['labels']
    encoder_mask = batch['encoder_mask']
    decoder_mask = batch['decoder_mask']
    
    # make sure the encoder and decoder are in training mode so dropout is applied
    transformer.train()
    
    assert encoder_inputs.size(0) == decoder_inputs.size(0)
    
    batch_size = encoder_inputs.size(0)
    
    #TODO pass the encoder_inputs through the encoder
    
    #TODO pass the decoder inputs and encoder outputs through the decoder
    
    #TODO get the predictions from the decoder outputs
    
    #TODO shift class dimenssion to be second for loss computation
    
    loss = criterion(predictions, labels)
    
    #update the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() 

######################################################################
    
def translate(transformer: Transformer, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    Translate a sentence using the transformer model with greedy decoding.
    """
    transformer.eval()
    
    with torch.no_grad():
        batch_size = 1
        
        # Prepare input
        input_tensor = tensor_from_sentence(src_vocab, sentence).permute(1, 0).to(device)
        
        # Pad input if necessary
        padding_length = max_length - input_tensor.size(1)
        if padding_length > 0:
            padding = torch.full((1, padding_length), PAD_index, dtype=input_tensor.dtype, device=input_tensor.device)
            input_tensor = torch.cat((input_tensor, padding), dim=1)
            
        # Create input mask and properly broadcast it
        input_mask = torch.cat([
            torch.ones(input_tensor.size(0), input_tensor.size(1) - padding_length),
            torch.zeros(input_tensor.size(0), padding_length)
        ], dim=1).bool().to(device)
        
        # Add necessary dimensions for attention
        input_mask = input_mask.unsqueeze(1).unsqueeze(2)
        
        # Encode the input sentence
        encoder_outputs, encoder_attentions = transformer.encode(input_tensor, input_mask)
        
        # Initialize decoder input with SOS token
        decoder_input = torch.full((batch_size, 1), SOS_index, dtype=torch.long, device=device)
        
        output_tokens = []
        
        # Greedy decoding
        for i in range(max_length):
            # Create target mask
            tgt_mask = create_causal_mask(decoder_input.size(1)).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            
            # Create padding mask for decoder input
            decoder_padding_mask = (decoder_input != PAD_index).unsqueeze(1).unsqueeze(2)
            
            # Combine padding mask with subsequent mask
            combined_mask = decoder_padding_mask & tgt_mask
            
            # Decode
            decoder_output, mask_attentions, cross_attentions = transformer.decode(
                decoder_input,
                encoder_outputs,
                input_mask,
                combined_mask
            )
            
            # Get prediction for next token
            predictions = transformer.get_predictions(decoder_output[:, -1:])
            _, next_token = predictions.max(dim=-1)
            
            # Add predicted token to output
            output_tokens.append(next_token.item())
            
            # Break if EOS token is predicted
            if next_token.item() == EOS_index:
                break
                
            # Add predicted token to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Convert indices to words
        output_words = []
        for idx in output_tokens:
            if idx == EOS_index:
                break
            output_words.append(tgt_vocab.index2word[idx])
            
        return output_words, encoder_attentions, mask_attentions, cross_attentions 
        
######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(transformer, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    """Translate a set of sentences with progress bar."""
    # Determine how many sentences to translate
    num_sentences = len(pairs)
    if max_num_sentences is not None:
        num_sentences = min(num_sentences, max_num_sentences)
    
    output_sentences = []
    
    # Create progress bar
    progress_bar = tqdm(
        total=num_sentences,
        desc='Translating sentences',
        position=0,
        leave=True
    )
    
    for pair in pairs[:num_sentences]:
        output_words, _, _, _ = translate(transformer, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
        
        # Update progress bar
        progress_bar.update(1)
        # Optionally show the current sentence pair
        progress_bar.set_postfix({
            'src': pair[0][:30] + '...' if len(pair[0]) > 30 else pair[0]
        })
    
    progress_bar.close()
    return output_sentences

######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(transformer, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _, _, _ = translate(transformer, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

######################################################################

def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())

######################################################################

def main():
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--batch_size', default=32, type=int,
                    help='batch size for training')
    
    #model parameters
    ap.add_argument('--embedding_size', default=128, type=int,
                    help='embedding size of transformer, also word vector size')
    ap.add_argument('--n_heads', default=4, type=int,
                    help='number of attention heads')
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of transformer, also word vector size')    
    ap.add_argument('--n_layers', default=2, type=int,
                    help='number of layers in transformer (same used for encoder and decoder)')
    
    #logging parameters
    ap.add_argument('--print_every', default=1, type=int,
                    help='print loss info every this many epochs examples')
    ap.add_argument('--checkpoint_every', default=1, type=int,
                    help='write out checkpoint every this many epochs examples')
    ap.add_argument('--bleu_every', default=3, type=int,
                    help='compute dev BLEU score every this many epochs examples')
    
    #learning parameters
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--n_epochs', default=10, type=int,
                help='total number of epochs to train on')
    
    #data parameters
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    
    ap.add_argument('--attentions_through_time', type=str, default=None,
                    help='Provide a string of a sentence and the program will save the attentions weights at the end of each epoch')
    
    args = ap.parse_args()
    
    # if we are tracking attention weights through time setup lists to track
    if args.attentions_through_time is not None:
        output_words_through_time = []
        encoder_attentions_through_time = []
        mask_attentions_through_time = []
        cross_attentions_through_time = []
    
        source_sentence = args.attentions_through_time

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        epoch_number = state['epoch_number']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        epoch_number = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)
    
    transformer = Transformer(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        embedding_size=args.embedding_size,
        n_heads=args.n_heads,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout_p=0.2).to(device)
    
    num_model_params = sum(p.numel() for p in transformer.parameters())
    logging.info('number of model parameters: ' + str(num_model_params))

    # transformer weights are randomly initialized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        transformer.load_state_dict(state['transformer_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(transformer.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_index, label_smoothing=0.1)

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every
    
    train_dataloader = create_dataloader(train_pairs, src_vocab, tgt_vocab, batch_size=args.batch_size)
    
    #translate the source sentence through time before any updates to see randomly weighted attention
    if args.attentions_through_time is not None:
        output_words, encoder_attentions, mask_attentions, cross_attentions = translate(transformer, source_sentence, src_vocab, tgt_vocab)
        output_words_through_time.append(output_words)
        encoder_attentions_through_time.append(encoder_attentions)
        mask_attentions_through_time.append(mask_attentions)
        cross_attentions_through_time.append(cross_attentions)
    
    #train the batch an epoch
    while epoch_number < args.n_epochs:
        # Create progress bar for each epoch
        progress_bar = tqdm(train_dataloader, 
                          desc=f'Epoch {epoch_number + 1}/{args.n_epochs}',
                          position=0, 
                          leave=True)
        
        transformer.train()  # Set model to training mode
        epoch_loss = 0
        
        # Train the batches for one epoch
        for batch_idx, batch in enumerate(progress_bar):
            
            loss = train_batch(batch, transformer, optimizer, criterion)
            print_loss_total += loss
            epoch_loss += loss
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
        
        # Close progress bar for the epoch
        progress_bar.close()
        
        if args.attentions_through_time is not None:
            output_words, encoder_attentions, mask_attentions, cross_attentions = translate(transformer, source_sentence, src_vocab, tgt_vocab)
            output_words_through_time.append(output_words)
            encoder_attentions_through_time.append(encoder_attentions)
            mask_attentions_through_time.append(mask_attentions)
            cross_attentions_through_time.append(cross_attentions)
            
        #after each epoch, print out some information
        if epoch_number % args.checkpoint_every == 0:
            state = {'epoch_number': epoch_number,
                     'transformer_state': transformer.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % epoch_number
            torch.save(state, filename)

        if epoch_number % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (epoch:%d iter/n_iters:%d%%) epoch_loss:%.4f',
                         time.time() - start,
                         epoch_number,
                         epoch_number / args.n_epochs,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(transformer, dev_pairs, src_vocab, tgt_vocab, n=5)
            
        if epoch_number %args.bleu_every == 0:
            translated_sentences = translate_sentences(transformer, dev_pairs, src_vocab, tgt_vocab)
            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)
            
        epoch_number += 1
        
    translated_sentences = translate_sentences(transformer, dev_pairs, src_vocab, tgt_vocab)    
    references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
    candidates = [clean(sent).split() for sent in translated_sentences]
    dev_bleu = corpus_bleu(references, candidates)
    logging.info('Training complete, final Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(transformer, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')
            
    if args.attentions_through_time is not None:
        #save data through time to file
        with open('output_words_through_time.pkl', 'wb') as f:
            pickle.dump(output_words_through_time, f)
            
        with open('encoder_attentions_through_time.pkl', 'wb') as f:
            pickle.dump(encoder_attentions_through_time, f)
            
        with open('mask_attentions_through_time.pkl', 'wb') as f:
            pickle.dump(mask_attentions_through_time, f)
            
        with open('cross_attentions_through_time.pkl', 'wb') as f:
            pickle.dump(cross_attentions_through_time, f)
    
if __name__ == '__main__':
    main()