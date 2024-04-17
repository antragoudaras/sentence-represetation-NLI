import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import VocabularyBuilder, DataLoaderBuilder
from encoders import BaselineEnc, UniDirLSTM
from classifier import Clasiifier
from model import Model
from train_procedure import train, evaluate
import os

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
    logging.info("Building/Loading the SNLI dataset...")

    log_dir = "tensorboard_log_dir"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    best_model_dir = "best_model_dir"
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)

    vocabulary_builder = VocabularyBuilder()
    dataset, w2i, embeddings_matrix = vocabulary_builder.build_vocabulary()

    dataloader = DataLoaderBuilder(dataset, w2i, args)
    train_loader, val_loader = dataloader.get_dataloader("train"), dataloader.get_dataloader("validation")
    test_loader = dataloader.get_dataloader("test")
    if args.encoder == "baseline":
        encoder = BaselineEnc(embeddings_matrix)
        classifier_dim = 300
    elif args.encoder == "unilstm":
        encoder = UniDirLSTM(embeddings_matrix)
        classifier_dim = 2048
    else:
        raise ValueError("Invalid encoder type")
    
    classifier = Clasiifier(input_dim=classifier_dim)

    model = Model(encoder, classifier).to(device)
    logging.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda= lambda epoch: 0.9, verbose=True)
    criterion = nn.CrossEntropyLoss()

   
    logging.info("Training the model...")
    best_val_loss, best_val_acc = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, args.num_epochs, log_dir, best_model_dir, args.encoder)
    logging.info(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.4f}")

    #Load the best model
    logging.info("Loading the best model...")
    model.load_state_dict(torch.load(os.path.join(best_model_dir, f"{args.encoder}_best.pth")))

    test_loss, test_acc = evaluate(model, criterion, test_loader, device)

    logging.info(f"Test loss: {test_loss:.4f}, Test accuracy: {100*test_acc:.4f}")
    logging.info("Done!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--encoder", type=str, default="baseline", help="Encoder type", choices=["baseline", "unilstm"])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s: %(message)s",
                        datefmt="%m/%d %I:%M:%S %p")
    
    main(args)