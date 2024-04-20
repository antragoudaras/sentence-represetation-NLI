import os 
import torch.nn as nn
import torch
import numpy as np
import logging
import argparse
from SentEval import senteval

from torch.nn.utils.rnn import pad_sequence
from data_utils import Tokenizer, VocabularyBuilder, DataLoaderBuilder
from encoders import BaselineEnc, UniLSTM, BiLSTM
from classifier import Clasiifier
from model import Model
from train_procedure import evaluate

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def token_mapping(w2i: dict[str, int], tokens: list[str]) -> list[int]:
    """Convert a list of tokens to a list of indices.

    Args:
        tokens (list[str]): A list of tokens

    Returns:
        list[int]: A list of indices
    """
    return torch.tensor([w2i.get(token, 0) for token in tokens])

def batcher(params, batch):
    """Batcher of SentEval"""
    batch = [sent if sent != [] else ['.'] for sent in batch]
    encoder = params.model.encoder
    w2i = params.w2i

    #Tokenize the batch
    tokinzer_obj = Tokenizer()
    tokenized_sentences = [tokinzer_obj.tokenize(' '.join(sentence)) for sentence in batch]

    #get indices of the tokens
    sentence_tokens = [token_mapping(w2i, tokens) for tokens in tokenized_sentences]

    #pad the sequences
    padded_sentences = pad_sequence(sentence_tokens, batch_first=True, padding_value=1)
    lengths = torch.tensor([len(token) for token in sentence_tokens])

    #to device
    padded_sentences, lengths = padded_sentences.to(params.device), lengths.to(params.device)

    #compute the snt_embdgs with a forward pass of the enoder
    sentence_emdgs = encoder(padded_sentences, lengths)

    return sentence_emdgs.cpu().detach().numpy()

def calc_macro_micro_acc(tasks, results):
    tasks_with_dev_acc = [task for task in tasks if 'devacc' in results[task]]
    macro_acc = np.mean([results[task]['devacc'] for task in tasks_with_dev_acc])
    micro_score = np.sum([results[task]['ndev'] * results[task]['devacc'] for task in tasks_with_dev_acc])
    micro_score /= np.sum([results[task]['ndev'] for task in tasks_with_dev_acc])

    return macro_acc, micro_score

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
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)

    #Defining the Loss
    criterion = nn.CrossEntropyLoss()
    
    #Evaluate the model on SNLI dataset if the flag is set
    if args.snli:
        logging.info("Evaluating the model on SNLI dataset")
        dataloader = DataLoaderBuilder(dataset, w2i, args)
        val_loader, test_loader = dataloader.get_dataloader("validation"), dataloader.get_dataloader("test")
        
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)
        
        logging.info(f"Validation loss: {val_loss:.4f}, Validation accuracy: {100*val_acc:.4f}")
        logging.info(f"Test loss: {test_loss:.4f}, Test accuracy: {100*test_acc:.4f}")
    #Evaluate the model on SentEval tasks if the flag is set
    if args.senteval:
        logging.info("Evaluating the model on SentEval tasks")
        params = {'args': args, 'model': model, 'w2i': w2i, 'device': device, 'task_data_path': args.sent_eval_path}

        se = senteval.engine.SE(params, batcher)
        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
        results = se.eval(transfer_tasks)
        logging.info(f"Results on SentEval tasks: {results}")

        macro_acc, micro_score = calc_macro_micro_acc(transfer_tasks, results)

        logging.info(f"Macro accuracy: {100*macro_acc:.4f}")
        logging.info(f"Micro score: {100*micro_score:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model either on SNLI and/or the SentEval tasks")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--encoder", type=str, default="baseline", help="Encoder type", choices=["baseline", "unilstm", "bilstm", "bilstm-max"])
    
    parser.add_argument("--snli", action="store_true", help="Evaluate on SNLI dataset")
    parser.add_argument("--senteval", action="store_true", help="Evaluate on SentEval tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
    parser.add_argument("--sent_eval_path", type=str, default="downstream", help="Path to the SentEval data")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s: %(message)s",
                        datefmt="%m/%d %I:%M:%S %p")
    
    main(args)