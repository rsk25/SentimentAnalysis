from collections import Counter
from typing import List
import random

import torch
from torch.backends import cudnn
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import Vocab

SEED = 1234

MAX_VOCAB_SIZE = 25000


class TextDataset(Dataset):

    def __init__(self, data_list: List[str]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_batch(batch: List[str]):
    """
    :param batch: dataset iterator (list of samples)
    :return: dynamically padded batch (mini-batch tensors)
    """

    # process raw text from dataset iterators
    text_transform = lambda x: [train_vocab[token] for token in tokenizer(x)]
    label_transform = lambda x: 1 if x == 'pos' else 0

    label_list, text_list = list(), list()
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        tensor_text = torch.tensor(text_transform(_text))
        text_list.append(tensor_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=0.0)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):  # text: [sent len(pad included), batch size]
        embedded = self.embedding(text)  # embedded: [sent len, batch size, hidden dim]

        output, hidden = self.rnn(embedded)
        # output: [sent len, batch size, hidden dim] -- concatenation of hidden states
        # hidden: [1, batch size,hidden dim] -- final hidden state

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))


def batch_sampler(tok, train_list, batch_s):
    """
    To group texts with similar lengths together, we randomly create multiple 'pools',
    and each of them has a size of `batch_size*100`.
    Then, we sort the samples within the individual pool by length.
    :param
    tok: tokenizer
    train_list: training dataset
    batch_s: batch_size
    :return: pools of samples
    """
    indices = [(i, len(tok(s[1]))) for i, s in enumerate(train_list)]
    random.shuffle(indices)
    pooled_indices = list()
    # create pool of indices with similar length
    for i in range(0, len(indices), batch_s * 100):
        pooled_indices.extend(sorted(indices[i: i + batch_s * 100], key=lambda x: x[1]))
    pooled_indices = [x[0] for x in pooled_indices]
    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_s):
        yield pooled_indices[i:i + batch_s]


def count_params(m: nn.Module) -> int:
    """
    :param m: the model instance
    :return: number of parameters
    """
    return sum(torch.numel(p) for p in m.parameters() if p.requires_grad())


def binary_accuracy(pred, y) -> float:
    """
    :param pred: predictions made by model
    :param y: the true label
    :return: accuracy value (per batch)
    """
    rounded_pred = torch.round(torch.sigmoid(pred))
    correct = (rounded_pred == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()  # at the beginning of each batch, reset the parameters
        predictions = model(batch.text).squeeze(1)
        # By default, model predictions are [batch size, 1]
        # PyTorch expects the prediction inputs to our criterion function to be of size [batch size]
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    cudnn.deterministic = True

    # get tokenizer
    tokenizer = get_tokenizer('spacy', "en_core_web_sm")

    # split train/test set (output: dataset iterators)
    train_data, test_data = IMDB(split=('train', 'test'))

    # tokenization
    counter_train = Counter()  # record count of each word to truncate 'rare' words
    for (label, line) in train_data:
        counter_train.update(tokenizer(line))
    # create vocab
    train_vocab = Vocab(counter_train, max_size=MAX_VOCAB_SIZE, specials=('<unk>', '<pad>'))

    # split train/validation set
    train_data, valid_data = random_split(train_data, [17500, 7500])
    print('Number of training examples: {}'.format(len(train_data)))
    print('Number of testing examples: {}'.format(len(test_data)))
    print('Number of validation examples: {}'.format(len(valid_data)))

    # map token strings to numerical values
    train_itos = train_vocab.itos

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    train_dataloader = DataLoader(TextDataset(train_data), batch_size=batch_size,
                                   shuffle=True, collate_fn=collate_batch)

    INPUT_DIM = len(train_vocab)  # dimension of one-hot vectors(= the vocab size)
    EMBEDDING_DIM = 100  # size of dense word vectors
    HIDDEN_DIM = 256  # size of hidden states
    OUTPUT_DIM = 1  # number of classes (for binary classification, this can be 1 dim)

    # RNN instance
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

    print('Model has {count:,} trainable parameters'.format(count=count_params(model)))

    # Train model
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  # Stochastic gradient descent
    criterion = nn.BCEWithLogitsLoss()  # loss function: binary cross entropy with logits
    model.to(device)
    criterion = criterion.to(device)


