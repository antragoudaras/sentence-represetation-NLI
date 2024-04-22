import argparse
import logging
import torch 
import numpy as np
from data_utils import VocabularyBuilder, Tokenizer
from encoders import BaselineEnc, UniLSTM, BiLSTM
from classifier import Clasiifier
from model import Model
from torch.nn.utils.rnn import pad_sequence



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def token_mapping(w2i, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of indices.

        Args:
            w2i: word to index mapping
            tokens (list[str]): A list of tokens

        Returns:
            list[int]: A list of indices
        """
        return torch.tensor([w2i.get(token, 0) for token in tokens])


def predict_entailment(premise, hypothesis, model, w2i, device):
    labels_dict = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    tokenizer_obj = Tokenizer()
    model.eval()
    with torch.no_grad():
        premise_tokens = tokenizer_obj.tokenize(premise)
        hypothesis_tokens = tokenizer_obj.tokenize(hypothesis)

        premise_indices = token_mapping(w2i, premise_tokens)
        hypothesis_indices = token_mapping(w2i, hypothesis_tokens)

        premise_lengths = torch.tensor([len(premise_indices)])
        hypothesis_lengths = torch.tensor([len(hypothesis_indices)])

        padded_premise = pad_sequence([premise_indices], batch_first=True, padding_value=1)
        padded_hypothesis = pad_sequence([hypothesis_indices], batch_first=True, padding_value=1)

        padded_premise, padded_hypothesis = padded_premise.to(device), padded_hypothesis.to(device)
        premise_lengths, hypothesis_lengths = premise_lengths.unsqueeze(0).to(device), hypothesis_lengths.unsqueeze(0).to(device)

        loggits = model(padded_premise, premise_lengths, padded_hypothesis, hypothesis_lengths)
        
        label_prediction = torch.argmax(loggits, dim=-1).item()
        
        return labels_dict[label_prediction]
        
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Printing arguments : {args}")
    logging.info("Setting seed...")
    set_seed(args.seed)

    vocabulary_builder = VocabularyBuilder()
    _, w2i, embeddings_matrix = vocabulary_builder.build_vocabulary()

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

    entailment = predict_entailment(args.premise, args.hypothesis, model, w2i, device)

    logging.info(f"Premise: {args.premise}")
    logging.info(f"Hypothesis: {args.hypothesis}")
    logging.info(f"Entailment prediction: {entailment}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Entailment on new text, using the trained model")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--encoder", type=str, default="bilstm-max", help="Encoder type", choices=["baseline", "unilstm", "bilstm", "bilstm-max"])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--premise", type=str, default="Two men sitting in the sun", help="Premise sentence")
    parser.add_argument("--hypothesis", type=str, default="Nobody is sitting in the shade", help="Hypothesis sentence")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s: %(message)s",
                        datefmt="%m/%d %I:%M:%S %p")
    
    main(args)