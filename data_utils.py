from spacy.lang.en import English
import logging 
import os
import torch
import torchtext
from torchtext.vocab import GloVe
from torch.utils.data import Dataset, DataLoader

class SNLIDataset():
    def __init__(self):
        self.tokenizer = English()

        self.TEXT = torchtext.data.Field(tokenize=self.tokenize, lower=True, include_lengths=True)
        self.LABEL = torchtext.data.Field(sequential=False, unk_token=None, is_target=True)

        print("Tokenizing SNLI dataset...")
        self.train, self.val, self.test = torchtext.datasets.SNLI.splits(self.TEXT, self.LABEL, root='data')
        
        vector_cache_loc = 'vector_cache/snli_vectors.pt'
        if os.path.isfile(vector_cache_loc):
            self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
        else:
            self.TEXT.build_vocab(self.train, vectors=GloVe(name='840B', dim=300))
            os.makedirs('vector_cache', exist_ok=True)
            torch.save(self.TEXT.vocab.vectors, vector_cache_loc)
        print('kippo')
    def tokenize(self, text):
        return [tok.text for tok in self.tokenizer(text)]
    
    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def get_vocab_size(self):
        return len(self.TEXT.vocab)

    def get_label_vocab(self):
        return self.LABEL.vocab.stoi


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    snli = SNLIDataset()
    print(snli.get_vocab_size())