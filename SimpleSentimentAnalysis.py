from collections import Counter

import torch
from torch.backends import cudnn
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB

SEED = 1234

if __name__ == "__main__":
    torch.manual_seed(SEED)
    cudnn.deterministic = True

    # get tokenizer
    tokenizer = get_tokenizer('spacy', "en_core_web_sm")

    # split train/test set
    train_data, test_data = IMDB(split=('train', 'test'))

    # tokenization
    for (label, line) in train_data:


    # split train/validation set
    train_data, valid_data = random_split(train_data, [17500, 7500])
    print('Number of training examples: {}'.format(len(train_data)))
    print('Number of testing examples: {}'.format(len(test_data)))
    print('Number of validation examples: {}'.format(len(valid_data)))


