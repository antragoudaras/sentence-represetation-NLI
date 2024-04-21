import argparse
import logging
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from spacy.lang.en import English
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

class Tokenizer:
    def __init__(self):
        self.nlp = English()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and lwercase a sentence
        Args:
            text (str): The sentence to tokenize

        Returns:
            list[str]: The list of tokens
        """
        return [token.text.lower() for token in self.nlp.tokenizer(text)]


class SentEvalVocabularyBuilder:
    def __init__(self, glove_version = '840B', word_embedding_dim = 300, tokenize=False):
        self.glove_version = glove_version
        self.word_embedding_dim = word_embedding_dim
        self.tokenize = tokenize
    def tokenize_batch(self, batch: dict) -> dict:
        """Tokenize the premise and hypothesis of a batch of examples.

        Args:
            batch (dict): A batch of examples from the dataset

        Returns:
            dict: A dictionary containing the tokenized premise and hypothesis
        """
        tokenizer = Tokenizer()
        return {
            "item": [tokenizer.tokenize(text) for text in batch],
        }

    def build_vocabulary(self, batch):
        """Build the vocabulary of the SNLI corpus.
        Args:
            split (str): The split of the dataset options are 'train', 'validation', or 'test'

        Returns:
            dataset (Dataset): SNLI dataset
            w2i (dict): mapping tokens to indices in the form of w2i
            aligned_embeddings (torch.Tensor): tensor of aligned embeddings
        """
        # skip the building process if the files already exist
        # if (Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists() and Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists()):
        #     logging.info("Vocab already exists. Loading from disk...")
        #     w2i = torch.load(
        #         f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt"
        #     )
        #     aligned_embeddings = torch.load(
        #         f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt"
        #     )
        #     dataset = self.get_dataset()
        #     return dataset, w2i, aligned_embeddings

        logging.info("Building vocabulary...")

        if self.tokenize:
            logging.info("Tokenizing the batch...")
            batch = self.tokenize_batch(batch)


        # Load the GloVe embeddings
        glove = GloVe(self.glove_version, dim=self.word_embedding_dim)

        # Create a dictionary mapping tokens to indices
        w2i = {"<UNK>": 0, "<PAD>": 1}

        #average the embeddings for the unknown token
        unk_embedding = glove.vectors.mean(dim=0)

        # Create a list of aligned embeddings
        aligned_embeddings = [unk_embedding, glove["<PAD>"]]

        # Only use the train split to build the vocabulary
        logging.info("Building unique tokens of vocab...")
        unique_tokens = {
            token
            for item in batch
            for token in item
        }

        #Sorting ensures alignment
        sorted_unique_tokens = sorted(unique_tokens)

        # Update the w2i dictionary and embeddings list
        logging.info("Building w2i and aligned embeddings...")
        for token in sorted_unique_tokens:
            w2i[token] = len(w2i)
            aligned_embeddings.append(glove[token])

        # Convert the list of aligned embeddings to a torch.Tensor
        aligned_embeddings = torch.stack(aligned_embeddings)

        # Save the token_to_idx dictionary and aligned_embeddings tensor to disk
        #make sure that the saved_vectors directory exists
        # Path("./saved_vectors").mkdir(exist_ok=True)
        # logging.info("Saving w2i and aligned embeddings to disk in folder saved_vectors...")
        # torch.save(
        #     w2i,
        #     f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt",
        # )
        # torch.save(
        #     aligned_embeddings,
        #     f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt",
        # )

        return batch, w2i, aligned_embeddings