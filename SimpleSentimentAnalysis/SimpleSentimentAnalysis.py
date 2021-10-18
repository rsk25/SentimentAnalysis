import sys
import time
from functools import partial

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab
from torch.backends import cudnn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

SEED = 1234

MAX_VOCAB_SIZE = 25000

WRITER = SummaryWriter()


def tokenize_data(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    length = len(tokens)
    return {'tokens': tokens, 'length': length}


def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}


def collate_batch(batch, pad_index):
    """
    :param
    batch: dataset iterator (list of samples)
    pad_index: the index for '<pad>'
    :return: dynamically padded batch (mini-batch tensors)
    """
    batch_ids = [b['ids'] for b in batch]
    batch_ids = pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)

    batch_len = [b['length'] for b in batch]
    batch_len = torch.stack(batch_len, dim=0)

    batch_label = [b['label'] for b in batch]
    batch_label = torch.stack(batch_label, dim=0)
    batch = {
        'ids': batch_ids,
        'length': batch_len,
        'label': batch_label
    }
    return batch


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, dropout=dropout_rate, batch_first=True, nonlinearity='relu')
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):  # text: [sent len(pad included), batch size]
        embedded = self.embedding(ids)  # embedded: [sent len, batch size, hidden dim]
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True,
                                                                  enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        hidden = self.dropout(hidden[-1])
        # packed_output: [batch size, seq len, hidden dim * n directions]
        # hidden: [batch size, hidden dim]
        prediction = self.linear(hidden)
        return prediction


def count_params(m) -> int:
    """
    :param m: the model instance
    :return: number of parameters
    """
    return sum(torch.numel(p) for p in m.parameters() if p.requires_grad)


def binary_accuracy(pred, y):
    """
    :param pred: predictions made by model
    :param y: the true label
    :return: accuracy value (per batch)
    """
    predicted_classes = pred.argmax(dim=-1)
    correct = predicted_classes.eq(y).sum()
    acc = correct / float(pred.shape[0])
    return acc


def train(dataloader, model, criterion, optimizer, device):
    epoch_losses = []
    epoch_accs = []
    model.train()

    for e, batch in enumerate(tqdm(dataloader, desc='Training...', file=sys.stdout)):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device, dtype=torch.float)
        prediction = model(ids, length).squeeze(1)
        loss = criterion(prediction, label)
        accuracy = binary_accuracy(prediction, label)
        WRITER.add_scalar("Loss/Train", loss, e)
        WRITER.add_scalar("Accuracy/Train", accuracy, e)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs


def evaluate(dataloader, model, criterion, device):
    epoch_losses = []
    epoch_accs = []
    model.eval()

    for e, batch in enumerate(tqdm(dataloader, desc='Evaluating...', file=sys.stdout)):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device, dtype=torch.float)
        prediction = model(ids, length).squeeze(1)
        loss = criterion(prediction, label)
        accuracy = binary_accuracy(prediction, label)
        WRITER.add_scalar("Loss/Test", loss, e)
        WRITER.add_scalar("Accuracy/Test", accuracy, e)

        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = elapsed_time // 60
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    torch.manual_seed(SEED)
    cudnn.deterministic = True

    # split train/test set (output: dataset iterators)
    train_data = datasets.load_dataset('imdb', split='train')
    test_data = datasets.load_dataset('imdb', split='test')

    # get tokenizer
    tokenizer = get_tokenizer('spacy', "en_core_web_sm")

    # tokenization
    max_length = 256
    train_data = train_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
    test_data = test_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

    # split train/validation set
    test_size = 0.25
    train_valid_data = train_data.train_test_split(test_size=test_size)
    train_data = train_valid_data['train']
    valid_data = train_valid_data['test']

    print('Number of training examples: {}'.format(len(train_data)))
    print('Number of testing examples: {}'.format(len(test_data)))
    print('Number of validation examples: {}'.format(len(valid_data)))

    # build vocab
    min_freq = 5  # words that do not appear under 5 times will be ignored
    special_tok = ['<unk>', '<pad>']
    vocab = torchtext.vocab.build_vocab_from_iterator(train_data['text'], min_freq=min_freq, specials=special_tok)

    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']

    vocab.set_default_index(unk_idx)  # token for OOV words

    train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
    test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

    train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    valid_data = valid_data.with_format(type='torch', columns=['ids', 'label', 'length'])
    test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length'])

    collate = partial(collate_batch, pad_index=pad_idx)

    batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VOCAB_SIZE = len(vocab)  # dimension of one-hot vectors(= the vocab size)
    EMBEDDING_DIM = 100  # size of dense word vectors
    HIDDEN_DIM = 256  # size of hidden states
    OUTPUT_DIM = 1  # number of classes (for binary classification, this can be 1 dim)
    DROPOUT_RATE = 0.5

    # RNN instance
    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT_RATE, pad_idx)

    print('Model has {count:,} trainable parameters'.format(count=count_params(model)))

    # Train model
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  # Stochastic gradient descent
    criterion = nn.BCEWithLogitsLoss()  # loss function: binary cross entropy with logits
    model = model.to(device)
    criterion = criterion.to(device)

    # save loss and accuracy values for each epoch (plotting purposes)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # train the model through multiple epochs
    N_EPOCHS = 5
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(train_dataloader, model, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        valid_losses.extend(valid_loss)
        valid_accs.extend(valid_acc)

        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), 'SSA1_model.pt')

        print(f'Epoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {epoch_valid_loss:.3f} |  Val. Acc: {epoch_valid_acc * 100:.2f}%')
        WRITER.flush()

    # measure test loss and accuracy
    model.load_state_dict(torch.load('SSA1_model.pt'))
    test_loss, test_acc = evaluate(test_dataloader, model, criterion, device)
    print(f'Test Loss: {test_loss[0]:.3f} | Test Acc: {test_acc[0] * 100:.2f}%')
    WRITER.close()
