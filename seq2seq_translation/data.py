import torch
import random

from collections import Counter

from torch.utils.data import DataLoader

from torchtext.datasets import Multi30k
from torchtext.vocab import vocab, Vocab
from torchtext.data.utils import get_tokenizer

from torch.nn.utils.rnn import pad_sequence


tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def get_token_index_or_unk(token: str, vcb: Vocab):
    if token in vcb.get_stoi():
        return vcb.get_stoi()[token]
    return vcb.get_stoi()['<unk>']

def transform_sequence(text: str, tokenizer, vcb: Vocab):
    return [vcb.get_stoi()['<bos>']] + [get_token_index_or_unk(token, vcb) for token in tokenizer(text)] + [vcb.get_stoi()['<eos>']]


def get_data_and_vocabs():
    train_data, val_data, test_data = Multi30k(split=('train', 'valid', 'test'))

    counter_en = Counter()
    counter_de = Counter()
    for (de, en) in train_data:
        counter_en.update(tokenizer_en(en))
        counter_de.update(tokenizer_de(de))

    vocab_en = vocab(counter_en, min_freq=5, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab_de = vocab(counter_de, min_freq=5, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    return (train_data, val_data, test_data), (vocab_en, vocab_de)


def get_batch_collate_fn(vocabs: tuple[Vocab, Vocab], tokenizers):

    seq_transformer_src = lambda text: transform_sequence(text, tokenizers[0], vocabs[0])
    seq_transformer_dst = lambda text: transform_sequence(text, tokenizers[1], vocabs[1])
    pad_indices = [vocabs[0].get_stoi()['<pad>'], vocabs[1].get_stoi()['<pad>']]

    def _collate_batch(batch: str):
        input_sequences, output_sequences = [], []
        for (_input_seq, _output_seq) in batch:
            processed_input_text = seq_transformer_src(_input_seq)
            processed_output_text = seq_transformer_dst(_output_seq)

            input_sequences.append(torch.tensor(processed_input_text))
            output_sequences.append(torch.tensor(processed_output_text))

        return pad_sequence(input_sequences, padding_value=pad_indices[0]), pad_sequence(output_sequences, padding_value=pad_indices[1])
    
    return _collate_batch


def bucket_batch_sampler(dataset, batch_size, tokenizer, pool_size=100):
    data_list = list(dataset)

    indices = [(i, len(tokenizer(s[0]))) for i, s in enumerate(data_list)]

    random.shuffle(indices)

    pooled_indices = []
    for i in range(0, len(indices), batch_size * pool_size):
        pooled_indices.extend(sorted(indices[i : i + batch_size*pool_size], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i : i+batch_size]


if __name__ == '__main__':
    (train_ds, val_ds, test_ds), (vocab_en, vocab_de) = get_data_and_vocabs()

    bucket_dataloader = DataLoader(
        list(train_ds), 
        batch_sampler=bucket_batch_sampler(train_ds, batch_size=8, tokenizer=tokenizer_de), 
        collate_fn=get_batch_collate_fn([vocab_de, vocab_en], [tokenizer_de, tokenizer_en])
    )

    for i, (source, target) in enumerate(bucket_dataloader):
        print('=== Input sequence  batch:')
        print(source)

        print('=== Output sequence batch:')
        print(target)

        print('===========================\n')

        if not i < 5:
            break
