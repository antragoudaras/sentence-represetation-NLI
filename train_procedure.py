import torch
import logging
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import os

def evaluate(model, criterion, loader, device):
    val_loss = 0.
    correct_pred = 0

    model.eval()

    with torch.no_grad():
        valiter = iter(loader)
        for batch_idx in range(len(loader)):
            batch_data = next(valiter)
            premises_tensors, hypotheses_tensors, premises_lengths, hypotheses_lengths, labels = batch_data

            premises_tensors, premises_lengths = premises_tensors.to(device), premises_lengths.to(device)
            hypotheses_tensors, hypotheses_lengths = hypotheses_tensors.to(device), hypotheses_lengths.to(device)
            labels = labels.to(device)

            output_loggits = model(premises_tensors, premises_lengths, hypotheses_tensors, hypotheses_lengths)

            loss = criterion(output_loggits, labels)

            val_loss += loss.item()

            correct_pred += (output_loggits.argmax(dim=1) == labels).sum().item()

            batch_acc = (output_loggits.argmax(dim=1) == labels).sum().item() / labels.shape[0]


            if batch_idx % 10 == 0:
                logging.info(f'Batch: {batch_idx+1}/{len(loader)}, Loss: {val_loss:.4f}, Acc: {100*batch_acc:.4f}')
        
        val_loss = val_loss/len(loader)
        val_acc = correct_pred/len(loader.dataset)
            
        return val_loss, val_acc

def train_step(model, opt, criterion, batch_data, device):
    premises_tensors, hypotheses_tensors, premises_lengths, hypotheses_lengths, labels = batch_data
    
    premises_tensors, premises_lengths = premises_tensors.to(device), premises_lengths.to(device)
    hypotheses_tensors, hypotheses_lengths = hypotheses_tensors.to(device), hypotheses_lengths.to(device)
    labels = labels.to(device)

    opt.zero_grad()

    output_loggits = model(premises_tensors, premises_lengths, hypotheses_tensors, hypotheses_lengths)

    loss = criterion(output_loggits, labels)

    loss.backward()

    opt.step()

    return loss


def train(model, optmizer, scheduler, criterion, train_loader, val_loader, device, epochs, lr_divisor, log_dir, best_model_dir, encoder_name):
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    best_val_acc = 0.



    for epoch in tqdm(range(epochs)):
        logging.info(f'Epoch: {epoch+1}/{epochs}')
        
        train_loss = 0.
        train_loss_per_batch = {}
        
        trainiter = iter(train_loader)

        model.train()

        for batch_idx in range(len(train_loader)):
            batch = next(trainiter)
            loss = train_step(model, optmizer, criterion, batch, device)
            train_loss += loss.item()
            train_loss_per_batch[batch_idx] = loss.item()

            if batch_idx % 500 == 0 and batch_idx != 0:
                logging.info(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Train Loss: {train_loss_per_batch[batch_idx]:.4f}')

        train_loss = train_loss /  len(train_loader)
        logging.info(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
        writer.add_scalar('Loss/train', train_loss, epoch)


        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        logging.info(f'Epoch: {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.4f}')
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', 100*val_acc, epoch)

        scheduler.step()

        if val_acc > best_val_acc:
            logging.info(f'Best model found with val acc.: {100*val_acc:.4f}')
            best_val_acc = val_acc
            model_pth = os.path.join(best_model_dir, f'{encoder_name}_best.pth')
            torch.save(model.state_dict(), model_pth)
        else:
            #divide the learning rate by the lr_divisor
            optmizer.param_groups[0]['lr'] /= lr_divisor
            logging.info(f'Learning rate decreased to: {optmizer.param_groups[0]["lr"]:.4f}')
            #stop training if the learning rate goes smaller than 10^-5
            if optmizer.param_groups[0]['lr'] < float(1e-5):
                logging.info('Learning rate is smaller than 10^-5, stopping the training...')
                break

    logging.info(f'Best val loss: {best_val_loss:.4f}')
    logging.info(f'Best val acc: {100*best_val_acc:.4f}')

    writer.flush()
    writer.close()

    return best_val_loss, best_val_acc