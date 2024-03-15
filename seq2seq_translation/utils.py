import torch 

from torchtext.data.metrics import bleu_score

from data import tokenizer_de, transform_sequence, bucket_batch_sampler

def translate_sentence(model, sentence: str, vocab_en, vocab_de, max_length=450):
    input_sequence = transform_sequence(sentence, tokenizer_de, vocab_de)

    sentence_tensor = torch.LongTensor(input_sequence).unsqueeze(1)  # .to(device) if I had a GPU

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [vocab_en.get_stoi()['<bos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]])  # .to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()  # this is where LLMs use the `temperature` arg, instead of picking the most likely next-word

        outputs.append(best_guess)  

        if best_guess == vocab_en.get_stoi()['<eos>']:
            break  

    
    translated_sentence = [vocab_en.get_itos()[idx] for idx in outputs]

    return translated_sentence[1:]


def save_checkpoint(state, filename="s2s_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename="s2s_checkpoint.pth.tar"):
    checkpoint = torch.load(filename)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])