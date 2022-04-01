from itertools import dropwhile
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

multiple_gpu = False


class WordBag(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, mode="sum"):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        """
        super(WordBag, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.EmbeddingBag(vocab_size, emb_size, mode=mode)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        # Get word embeddings, apply dropout
        # sentences = torch.nn.utils.rnn.pack_padded_sequence(sentences, words_per_sentence.cpu(), batch_first=True, enforce_sorted=False)
        # print("Before Word Embedding Shape {}".format(sentenceWordBags.shape))
        sentences = self.embeddings(sentences)  # (n_sentences, word_pad_len, emb_size)
        # print("After word Embedding shape {}".format(sentences.shape))
        return sentences

class WordLSTMBag(nn.Module):
      def __init__(self, vocab_size, emb_size, hidden_size, padding_index = 0, rnn_model='LSTM' , bidirectional = True):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        """
        super(WordLSTMBag, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = WordLSTM(vocab_size, emb_size, hidden_size, padding_index = 0, rnn_model='LSTM' , bidirectional = True)
        self.embeddings_bag = WordBag(vocab_size, emb_size, mode="sum")

      def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.init_embeddings(embeddings)
        self.embeddings_bag.init_embeddings(embeddings)
    
      def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        self.embeddings.fine_tune_embeddings(fine_tune)
        self.embeddings_bag.fine_tune_embeddings(fine_tune)

      def forward(self, x):
          out1 = self.embeddings(x)
          out2 = self.embeddings_bag(x)
          out = torch.cat( (out1, out2), dim=1)
          return out

class SelfAttentionAggregator(nn.Module):
    def __init__(self):
        pass

        

class WordLSTM(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, hidden_size, padding_index = 0, rnn_model='LSTM', dropout_ratio=0.2 , bidirectional = True):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        """
        super(WordLSTM, self).__init__()

        # Embeddings (look-up) layer
        #self.dropout = nn.Dropout(dropout_ratio)
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.rnn_model = rnn_model
       # self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        if rnn_model == "LSTM":
            self.rnn = nn.LSTM( input_size=emb_size, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True, bidirectional=bidirectional)
        elif rnn_model == "GRU":
            self.rnn = nn.GRU( input_size=emb_size, hidden_size=hidden_size, num_layers=1,
                                batch_first=True, bidirectional=bidirectional )
        else:
            raise "only support LSTM and GRU"

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)
       

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune
        
    def forward(self, sentences_x, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        # Get word embeddings, apply dropout
        #print("After word Embedding shape {}".format(sentences.shape))
        #print(sentences_x.shape)
        sentences = self.embeddings(sentences_x)  # (n_sentences, word_pad_len, emb_size)
        #print(sentences.shape)
        #print("After word Embedding shape {}".format(sentences.shape))
        packed_input = pack_padded_sequence(sentences, words_per_sentence.cpu(), batch_first=True, enforce_sorted=False)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        if self.rnn_model == "LSTM":
            _, ( ht, _ ) = self.rnn(packed_input)
        elif self.rnn_model == "GRU":
            _, ht = self.rnn(packed_input)
        else:
            raise "None RNN Model"
        #print("ht shape {}".format(ht.shape))
        #output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        #print(ht[-1].shape)
        #print(sentences_bag.shape)
        #print(out.shape)
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        
        ht = torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1)

       
        return ht


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(
        self,
        vocab_size,
        emb_size,
        word_rnn_size,
        word_rnn_layers,
        word_att_size,
        dropout,
        padding_index
    ):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings =  nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(
            emb_size,
            word_rnn_size,
            num_layers=word_rnn_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )

        # Word-level attention network
        self.word_attention = nn.Linear( word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """
        #print(sentences.shape)
        # Get word embeddings, apply dropout
        sentences = self.dropout(
            self.embeddings(sentences)
        )  # (n_sentences, word_pad_len, emb_size)

        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            sentences,
            lengths=words_per_sentence.tolist(),
            batch_first=True,
            enforce_sorted=False,
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words
        )  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = (
            att_w.max()
        )  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        if multiple_gpu:
            total_length = packed_words.data.size(1)
        else:
            total_length = None
        att_w, _ = pad_packed_sequence(
            PackedSequence(
                data=att_w,
                batch_sizes=packed_words.batch_sizes,
                sorted_indices=packed_words.sorted_indices,
                unsorted_indices=packed_words.unsorted_indices,
            ),
            batch_first=True,
            total_length=total_length,
        )  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(
            att_w, dim=1, keepdim=True
        )  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(
            packed_words, batch_first=True, total_length=total_length
        )  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(
            2
        )  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)
        #print(sentences.shape)
        return sentences
