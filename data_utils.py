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
    def __init__(self, glove_version = '840B', word_embedding_dim = 300):
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

    def build_vocabulary(self):
        """Build the vocabulary of the SNLI corpus.
        Args:
            split (str): The split of the dataset options are 'train', 'validation', or 'test'

        Returns:
            dataset (Dataset): SNLI dataset
            w2i (dict): mapping tokens to indices in the form of w2i
            aligned_embeddings (torch.Tensor): tensor of aligned embeddings
        """
        # skip the building process if the files already exist
        if (Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists() and Path(f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt").exists()):
            logging.info("Vocab already exists. Loading from disk...")
            w2i = torch.load(
                f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt"
            )
            aligned_embeddings = torch.load(
                f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt"
            )
            dataset = self.get_dataset()
            return dataset, w2i, aligned_embeddings

        logging.info("Building vocabulary...")

        # Load the SNLI dataset
        dataset = self.get_dataset()

        # Load the GloVe embeddings
        glove = GloVe(self.glove_version, dim=self.word_embedding_dim)

        # Create a dictionary mapping tokens to indices
        w2i = {"<UNK>": 0, "<PAD>": 1}

        #average the embeddings for the unknown token
        unk_embedding = glove.vectors.mean(dim=0)

        # Create a list of aligned embeddings
        aligned_embeddings = [unk_embedding, glove["<PAD>"]]

        # Only use the train split to build the vocabulary
        unique_tokens = {
            token
            for item in dataset["train"]
            for token in item["premise"] + item["hypothesis"]
        }

        #Sorting helps with alignment
        sorted_unique_tokens = sorted(unique_tokens)

        # Update the w2i dictionary and embeddings list
        for token in sorted_unique_tokens:
            w2i[token] = len(w2i)
            aligned_embeddings.append(glove[token])

        # Convert the list of aligned embeddings to a torch.Tensor
        aligned_embeddings = torch.stack(aligned_embeddings)

        # Save the token_to_idx dictionary and aligned_embeddings tensor to disk
        #make sure that the saved_vectors directory exists
        Path("./saved_vectors").mkdir(exist_ok=True)
        torch.save(
            w2i,
            f"./saved_vectors/w2i_{self.glove_version}_{self.word_embedding_dim}.pt",
        )
        torch.save(
            aligned_embeddings,
            f"./saved_vectors/embeddings_{self.glove_version}_{self.word_embedding_dim}.pt",
        )

        return dataset, w2i, aligned_embeddings

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
    def __init__(self, dataset, w2i: dict[str, int], args: argparse.Namespace):
        self.dataset = dataset
        self.w2i = w2i
        self.args = args
  
    
    def token_mapping(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of indices.

        Args:
            tokens (list[str]): A list of tokens

        Returns:
            list[int]: A list of indices
        """
        return torch.tensor([self.w2i.get(token, 0) for token in tokens])

    def collate_fn(self, batch):
        """Collate function for the SNLI dataset.

        Inputs:
            batch of data from the SNLI dataset (premise, hypothesis, label)
        Returns:
            tuple: padded_premises, padded_hypotheses, premise_lengths, hypothesis_lengths, labels
        """
        # Separate premises, hypotheses, and labels
        premises, hypotheses, labels = zip(*[(item["premise"], item["hypothesis"], item["label"]) for item in batch])

        # Convert tokens to indices
        premises = [self.token_mapping(premise) for premise in premises]
        hypotheses = [self.token_mapping(hypothesis) for hypothesis in hypotheses]

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
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )  
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = argparse.Namespace(batch_size=32, num_workers=4)
    vocabulary_builder = VocabularyBuilder()
    dataset, w2i, embeddings = vocabulary_builder.build_vocabulary()
    dataloader_builder = DataLoaderBuilder(dataset, w2i, args)
    train_dataloader = dataloader_builder.get_dataloader("train")