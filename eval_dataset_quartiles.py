import os 
import torch.nn as nn
import torch
import numpy as np
import logging
import argparse
import sys


from torch.nn.utils.rnn import pad_sequence
from data_utils import Tokenizer, VocabularyBuilder, DataLoaderBuilder
from encoders import BaselineEnc, UniLSTM, BiLSTM
from classifier import Clasiifier
from model import Model
from train_procedure import evaluate
from data_utils_senteval import SentEvalVocabularyBuilder


class QuartileDataloaderBuilder(DataLoaderBuilder):
    def __init__(self, quartile: list, **kwargs):
        super(QuartileDataloaderBuilder, self).__init__(**kwargs)
        self.quartile = quartile

    def combine_length_calc(self, sample: dict):
        sample["avg_len"] = len(sample["premise"]) + len(sample["hypothesis"])
        return sample
    
    def quartile_dataset(self, dataset):
        #Sort the dataset based on the length of the premise and hypothesis
        dataset1 = dataset.map(self.combine_length_calc)
        dataset2 = dataset.map(self.combine_length_calc)
        
        #Quartile Calculation
        length = dataset1["avg_len"] #Get the length of the dataset
        quartiles = np.quantile(length, self.quartile) #Calculate the quartile

        #Filter the dataset based on the quartile
        dataset1 = dataset.map(self.combine_length_calc)
        
        
        
        quantiled_dataset = []
        for idx, quartile in enumerate(quartiles):
            if idx == 0:
                lower_bound = 0
            else:
                lower_bound = quartiles[idx-1]
            upper_bound = quartile
            #Append the quartile
            quantiled_dataset.append(dataset1.filter(lambda x: lower_bound <= x["avg_len"] <= upper_bound))
        
        #Append the last quartile
        quantiled_dataset.append(dataset1.filter(lambda x: quartiles[-1] <= x["avg_len"]))
        
        #Remove the length key
        for subset in quantiled_dataset:
            subset = subset.remove_columns("avg_len")
        
        
        
        # Filter the dataset based on the quartile
        quantiled_dataset2 = [
            dataset2.filter(lambda x: lower_bound <= x["avg_len"] <= upper_bound)
            for lower_bound, upper_bound in zip([0] + list(quartiles), quartiles)
        ]
        
        # Append the last quartile
        quantiled_dataset2.append(dataset2.filter(lambda x: quartiles[-1] <= x["avg_len"]))
        
        # Remove the length key
        quantiled_dataset2 = [subset.remove_columns("avg_len") for subset in quantiled_dataset2]


        return quantiled_dataset2
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Printing arguments : {args}")
    logging.info("Setting seed...")
    set_seed(args.seed)

    #Building the vocabulary based on SNLI dataset
    vocabulary_builder = VocabularyBuilder()
    dataset, w2i, embeddings_matrix = vocabulary_builder.build_vocabulary()

    #Loading the model checkpoint
    logging.info(f"Loading the model checkpoint from {args.checkpoint}")
    if args.encoder == "baseline":
        encoder = BaselineEnc(embeddings_matrix)
        classifier_dim = 300
    
    elif args.encoder == "unilstm":
        encoder = UniLSTM(embeddings_matrix)
        classifier_dim = 2048
    
    elif args.encoder == "bilstm":
        encoder = BiLSTM(embeddings_matrix, max_pooling=False)
        classifier_dim = 4096
    
    elif args.encoder == "bilstm-max":
        encoder = BiLSTM(embeddings_matrix, max_pooling=True)
        classifier_dim = 4096
    
    else:
        raise ValueError("Invalid encoder type")
    
    classifier = Clasiifier(input_dim=classifier_dim)

    model = Model(encoder, classifier).to(device)
    #Load the model checkpoint
    logging.info("Loading the model checkpoint trained in SNLI dataset")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    #Defining the Loss
    criterion = nn.CrossEntropyLoss()

    dataset_dict = {"dataset": dataset, "w2i": w2i, "args": args}
    quariles_list = args.quartile_list
    dataloader = QuartileDataloaderBuilder(quartile=quariles_list, **dataset_dict)
    test_loader = dataloader.get_dataloader("test")
    quantiled_dataset = dataloader.quartile_dataset(test_loader)
    logging.info(f"Shortest dataset size: {len(quantiled_dataset[0])}")
    logging.info(f"Middle dataset size: {len(quantiled_dataset[1])}")
    logging.info(f"Longest dataset size: {len(quantiled_dataset[2])}")

    logging.info("Evaluating the model on SNLI dataset")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Evaluate the model either on SNLI and/or the SentEval tasks")
        parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
        parser.add_argument("--encoder", type=str, default="bilstm-max", help="Encoder type", choices=["baseline", "unilstm", "bilstm", "bilstm-max"])
        parser.add_argument("--seed", type=int, default=1111, help="Random seed")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument("--quartile_list", type=list, default=[0.33, 0.66], help="Quartile list")
        args = parser.parse_args()

        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s: %(message)s",
                            datefmt="%m/%d %I:%M:%S %p")
        
        main(args)
    