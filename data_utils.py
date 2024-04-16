"""This module contains the code for building the dataset and helper functions."""

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



class VocabularyBuilder:
    def __init__(self, glove_version: str = '840B', word_embedding_dim: int = 300):
        self.glove_version = glove_version
        self.word_embedding_dim = word_embedding_dim

    def tokenize_batch(self, batch: dict) -> dict:
        """Tokenize the premise and hypothesis of a batch of examples.

        Args:
            batch (dict): A batch of examples from the dataset

        Returns:
            dict: A dictionary containing the tokenized premise and hypothesis
        """
        tokenizer = Tokenizer()
        return {
            "premise": [tokenizer.tokenize(text) for text in batch["premise"]],
            "hypothesis": [tokenizer.tokenize(text) for text in batch["hypothesis"]],
        }

    def build_vocabulary(self) -> tuple[dict[str, int], torch.Tensor]:
        """Build the vocabulary of the SNLI corpus.
        Args:
            split (str): The split of the dataset options are 'train', 'validation', or 'test'

        Returns:
            dataset (Dataset): The SNLI dataset
            token_to_idx (dict): A dictionary mapping tokens to indices in the form of w2i
            aligned_embeddings (torch.Tensor): A tensor of aligned embeddings
        """
        # skip the building process if the files already exist
        if (Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists() and Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists()):
            logging.info("Vocab already exists. Loading from disk...")
            token_to_idx = torch.load(
                f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt"
            )
            aligned_embeddings = torch.load(
                f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt"
            )
            dataset = self.get_dataset()
            return dataset, token_to_idx, aligned_embeddings

        logging.info("Building vocabulary...")

        # Load the SNLI dataset
        dataset = self.get_dataset()

        # Load the GloVe embeddings
        glove = GloVe(self.glove_version, dim=self.word_embedding_dim)

        # Create a dictionary mapping tokens to indices
        token_to_idx = {"<UNK>": 0, "<PAD>": 1}

        #average the embeddings for the unknown token
        unk_embedding = glove.vectors.mean(dim=0)

        # Create a list of aligned embeddings
        aligned_embeddings = [unk_embedding, glove["<PAD>"]]

        # Get unique tokens from the dataset
        unique_tokens = {
            token
            for split in dataset.keys()
            for item in dataset[split]
            for token in item["premise"] + item["hypothesis"]
        }

        # Sort the unique tokens so that the indices are aligned the same way every time
        sorted_unique_tokens = sorted(unique_tokens)

        # Update the token_to_idx dictionary and aligned_embeddings list
        for token in sorted_unique_tokens:
            token_to_idx[token] = len(token_to_idx)
            aligned_embeddings.append(glove[token])

        # Convert the list of aligned embeddings to a torch.Tensor
        aligned_embeddings = torch.stack(aligned_embeddings)

        # Save the token_to_idx dictionary and aligned_embeddings tensor to disk
        #make sure that the saved_vectors directory exists
        Path("./saved_vectors").mkdir(exist_ok=True)
        torch.save(
            token_to_idx,
            f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt",
        )
        torch.save(
            aligned_embeddings,
            f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt",
        )

        return dataset, token_to_idx, aligned_embeddings

    def get_dataset(self) -> Dataset:
        """Get the dataset and tokenize the premise and hypothesis.

        Returns:
            Dataset: Tokenized dataset
        """
    
        dataset = load_dataset("snli", cache_dir="./.data")
        dataset = dataset.filter(lambda example: example["label"] != -1)
        dataset = dataset.map(self.tokenize_batch, batched=True, batch_size=1024)

        return dataset

class DataLoaderBuilder:
    def __init__(self, dataset, token_to_idx: dict[str, int], args: argparse.Namespace):
        self.dataset = dataset
        self.token_to_idx = token_to_idx
        self.args = args
  
    
    def tokens_to_indices(self, tokens: list[str], token_to_idx: dict[str, int]) -> list[int]:
        """Convert a list of tokens to a list of indices.

        Args:
            tokens (list[str]): A list of tokens
            token_to_idx (dict[str, int]): A dictionary mapping tokens to indices

        Returns:
            list[int]: A list of indices
        """
        return torch.tensor([token_to_idx.get(token, 0) for token in tokens])

    def collate_fn(self, token_to_idx, batch):
        """Collate function for the SNLI dataset.

        Args:
            token_to_idx (dict): A dictionary mapping tokens to indices
            batch (list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A batch
                of data from the SNLI dataset (premise_indices, hypothesis_indices, label)
        """
        # Separate premises, hypotheses, and labels
        premises, hypotheses, labels = zip(*[(item["premise"], item["hypothesis"], item["label"]) for item in batch])

        # Convert tokens to indices
        premises = [self.tokens_to_indices(premise, token_to_idx) for premise in premises]
        hypotheses = [self.tokens_to_indices(hypothesis, token_to_idx) for hypothesis in hypotheses]

        # Compute lengths
        premise_lengths = torch.tensor([len(premise) for premise in premises])
        hypothesis_lengths = torch.tensor([len(hypothesis) for hypothesis in hypotheses])

        # Pad sequences in premises and hypotheses using pad_sequence
        padded_premises = pad_sequence(premises, batch_first=True, padding_value=1)
        padded_hypotheses = pad_sequence(hypotheses, batch_first=True, padding_value=1)

        # Convert labels to tensor
        labels = torch.tensor(labels)

        return (
            padded_premises,
            padded_hypotheses,
            premise_lengths,
            hypothesis_lengths,
            labels
        )
    
    def get_dataloader(self, split: str) -> DataLoader:
        """Get the validation dataloader.

        Args:
            split (str): The split to use

        Returns:
            DataLoader: The validation dataloader
        """

        return DataLoader(
            self.dataset[split],
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=partial(self.collate_fn, self.token_to_idx),
            num_workers=self.args.num_workers,
        )  
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = argparse.Namespace(batch_size=32, num_workers=4)
    vocabulary_builder = VocabularyBuilder()
    dataset, token_to_idx, aligned_embeddings = vocabulary_builder.build_vocabulary()
    dataloader_builder = DataLoaderBuilder(dataset, token_to_idx, args)
    train_dataloader = dataloader_builder.get_dataloader("train")