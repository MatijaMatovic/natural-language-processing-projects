import random
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import Counter

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import vocab, Vocab

tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')


class Encoder(nn.Module):
    def __init__(self, 
                 input_size, embedding_size, hidden_size, 
                 num_layers, dropout_p, attention=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=hidden_size, 
            bidirectional=True if attention else False,  # i know i could have just sad =attention but this is more expressive i guess
            num_layers=num_layers, 
            dropout=dropout_p  # !!! MUST BE 0 (ZERO) FOR BIDIR
        )
        
        # If the model uses attention, the LSTM is bidirecitonal.
        # This means the dimensionality of outputs (hidden, cell) is doubled
        # They need to be combined somehow
        # Some approaches just add up the two directions 
        # Some do bitwise multiplication
        # Here we learn how to combine them with an FC layer
        self.attention_flatten_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.attention_flatten_cell = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, x):
        # x shape: (seq_length, batch_size,) 
        # embedding shape: (seq_length, batch_size, embedding_size) - turns the each word ID to some latent representation
        embedding = self.dropout(self.embedding(x))
        encoder_states, (hidden, cell) = self.rnn(embedding)

        if self.attention:
            hidden = torch.hstack(hidden[0], hidden[1]).unsqueeze(dim=0)
            cell = torch.hstack(cell[0], cell[1]).unsqueeze(dim=0)

            hidden = self.attention_flatten_hidden(hidden)
            cell = self.attention_flatten_cell(cell)

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, 
                 input_size, embedding_size, hidden_size, output_size, 
                 num_layers, dropout_p, attention=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)

        rnn_input_size = hidden_size * 2 + embedding_size if attention else hidden_size

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p  # !!! MUST BE 0 (ZERO) FOR BIDIR
        )

        # input size = enc_hidden_size (hidden_size * 2 for bidir + hidden_size of previous decoder step)
        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_states=None):
        # shape of x: (batch_size, ). Required (1, batch_size)
        x = x.unsqueeze(0)

        # embedding shape: (1, batch_size, embedding_size)
        embedding = self.dropout(self.embedding(x))

        if self.attention:
            sequence_length = encoder_states.shape[0]

            # repeat state to calculate it's codependence to each input sequence token
            # in one matrix multiplication
            h_reshaped = hidden.repeat(sequence_length, 1, 1)

            energy = self.relu(
                self.energy(
                    torch.hstack(h_reshaped, encoder_states).unsqueeze(dim=0)
                )
            )

            # (seq_length, batch_size, 1)
            attention = self.softmax(energy)

            # Doing permutation to obtain batch-first shape, for batch matmul to be possible
            attention = attention.permute(1, 2, 0)  # (seq_len, batch_size, 1) --> (batch_size, 1, seq_len)
            encoder_states = encoder_states.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size*2) --> (batch_size, seq_len, hidden_size*2)

            # batch matmul - how much each target token depends on each encoder state (source token)
            # then, undoing batch-first from the previous layer, after batch matmul
            # bmatmul: (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size*2) ==> (batch_size, 1, hidden_size*2)
            # permute: (batch_size, 1, hidden_size*2) --> (1, batch_size, hidden_size*2)
            context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

            embedding = torch.hstack((context_vector, embedding)).unsqueeze(0)

        # shape of outputs: (1, seq_length, hidden_size)
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        # shape of preds: (1, seq_length, vocab_size)
        # required: (seq_length, vocab_size)
        predictions = self.fc(output) 
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, decoder, 
                 source_vocab: Vocab, target_vocab: Vocab, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        self.encoder, self.decoder = encoder ,decoder
        self.source_vocab, self.target_vocab = source_vocab, target_vocab

    def forward(self, input_seq, target_seq, teacher_force_ratio=.5):
        # input_seq shape: (seq_len, batch_size)
        batch_size = input_seq.shape[1]
        target_len = target_seq.shape[0]

        outputs = torch.zeros(target_len, batch_size, len(self.target_vocab))  # .to(device), but i only have CPU so...
        encoder_states, hidden, cell = self.encoder(input_seq)

        x = target_seq[0]  # grab starting token

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_states)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target_seq[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
