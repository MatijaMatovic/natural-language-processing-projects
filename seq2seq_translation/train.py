import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Subset

# from torch.utils.tensorboard.writer import SummaryWriter

from data import (
    tokenizer_en,
    tokenizer_de,
    get_data_and_vocabs, 
    bucket_batch_sampler, 
    get_batch_collate_fn
)
from model import Encoder, Decoder, Seq2SeqClassic
from utils import load_checkpoint, save_checkpoint, translate_sentence

NUM_EPOCHS = 20
LEARNING_RAGE = 1e-3
BATCH_SIZE = 64

LOAD_MODEL = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ENCODER_EMBED_SIZE = 196
DECODER_EMBED_SIZE = 196
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
DROPOUT_PROB_ENC = .5
DROPOUD_PROB_DEC = .5

# WRITER = SummaryWriter(f'runs/loss_plot')
STEP = 0

(train_ds, val_ds, test_ds), (vocab_en, vocab_de) = get_data_and_vocabs()


encoder_input_size = len(vocab_de)
decoder_input_size = len(vocab_en)

output_size = len(vocab_en)

encoder = Encoder(encoder_input_size, ENCODER_EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB_ENC)
decoder = Decoder(decoder_input_size, DECODER_EMBED_SIZE, HIDDEN_SIZE, output_size, NUM_LAYERS, DROPOUD_PROB_DEC)

model = Seq2SeqClassic(encoder, decoder, vocab_de, vocab_en)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

pad_idx = vocab_en.get_stoi()['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if LOAD_MODEL:
    load_checkpoint(torch.load())


sentence = 'Es war Anfang Mai und, nach naßkalten Wochen, ein falscher Hochsommer eingefallen'
sentence_longer = 'Der Englische Garten, obgleich nur erst zart belaubt, war dumpfig wie im August und in der Nähe der Stadt voller Wagen und Spaziergänger gewesen'

train_dataloader = DataLoader(
    list(train_ds),
    batch_sampler=bucket_batch_sampler(train_ds, batch_size=8, tokenizer=tokenizer_de), 
    collate_fn=get_batch_collate_fn([vocab_de, vocab_en], [tokenizer_de, tokenizer_en])
)

for epoch in range(NUM_EPOCHS):
    print(f'[EPOCH {epoch} / {NUM_EPOCHS}]')

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = ' '.join(translate_sentence(model, sentence, vocab_en, vocab_de))
    # print(f'Example translation: \n {translated_sentence}')

    model.train()
    epoch_loss, batch_cnt = 0, 0
    loss_logg = tqdm(total=0, position=1, bar_format='{desc}')
    for batch_idx, (source, target) in tqdm(enumerate(train_dataloader), total=len(list(train_ds)) // BATCH_SIZE, position=2):
        output = model(source, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)

        # back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        epoch_loss += loss
        batch_cnt += 1
        if batch_idx % 10 == 0:
            loss_logg.set_description_str(f'Running epoch loss: {epoch_loss / batch_cnt}')
        # WRITER.add_scalar('Training loss', loss, global_step=STEP)

        STEP += 1

    print(f'Epoch loss: {epoch_loss / batch_cnt}')

