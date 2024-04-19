import torch.nn as nn
import torch 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch.autograd import Variable

class BaselineEnc(nn.Module):
    """Average word embeddings to obtain sentence representations. Take different lengths intop account."""
    def __init__(self, glove_embeddings):
        super(BaselineEnc, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.embeddings.requires_grad = False

    def forward(self, indices, lengths):
         
         embeddings = self.embeddings(indices)
         # Sum the embeddings along the sequence dimension (dim=1)
         sum_embeddings = torch.sum(embeddings, dim=1)
         # Compute the average by dividing the sum by the sequence lengths
         lengths = lengths.view(-1, 1).to(torch.float32)  # Ensure lengths have the same dtype as sum_embeddings
         avg_embeddings = sum_embeddings / lengths
         
         return avg_embeddings

class UniLSTM(nn.Module): 
    """Uni-directional LSTM encoder."""

    def __init__(self,  glove_embedding, input_dim=300, hid_dim=2048):
        super(UniLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embedding)
        self.embeddings.requires_grad = False

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True)

    def forward(self, indices, lengths):
        orig_emdgs = self.embeddings(indices).transpose(0, 1) # (seq_len, batch, input_size)

        #sort by lengts
        idx_sort = np.argsort(-lengths.cpu())
        idx_unsort = np.argsort(idx_sort)
        sent_len_sorted = -np.sort(-lengths.cpu())

        emdgs = orig_emdgs.index_select(1, Variable(idx_sort.cuda() if torch.cuda.is_available() else idx_sort))
        
        sent_len_sorted = torch.tensor(sent_len_sorted).to('cpu')

        packed_emdgs = pack_padded_sequence(emdgs, sent_len_sorted)


        (hidden_state, _) = self.lstm(packed_emdgs)[1]
        hidden_state = hidden_state.squeeze(0)

        final = hidden_state.index_select(0, Variable(idx_unsort.cuda() if torch.cuda.is_available() else idx_unsort))

        #return the last hidden state as the sentence representation
        return final
    
class BiLSTM(nn.Module):
    """Bi-directional LSTM encoder."""

    def __init__(self, glove_embedding, input_dim=300, hid_dim=2048, max_pooling=True):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embedding)
        self.embeddings.requires_grad = False

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True, bidirectional=True)
        self.max_pooling = max_pooling  # if True, use max pooling instead of concatenating the two hidden states

    def forward(self, indices, lengths):
        orig_emdgs = self.embeddings(indices).transpose(0, 1) # (seq_len, batch, input_size)

        #sort by lengts
        idx_sort = np.argsort(-lengths.cpu())
        idx_unsort = np.argsort(idx_sort)
        sent_len_sorted = -np.sort(-lengths.cpu())

        emdgs = orig_emdgs.index_select(1, Variable(idx_sort.cuda() if torch.cuda.is_available() else idx_sort))
        
        sent_len_sorted = torch.tensor(sent_len_sorted).to('cpu')

        packed_emdgs = pack_padded_sequence(emdgs, sent_len_sorted)

        packed_output, (hidden_states, _) = self.lstm(packed_emdgs)

        if self.max_pooling:
            lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]

            #unsort by length 
            lstm_out = lstm_out.index_select(0, Variable(idx_unsort.cuda() if torch.cuda.is_available() else idx_unsort)) #(batch_size, seq_len, hid_dim*2)

     
            #remove zero padding for max pooling
            # tensor_unpadded = [x[:l] for x, l in zip(lstm_out, lengths)] #list of length batch_size, each element is a tensor of shape (seq_len, hid_dim*2)
            # max = [torch.max(x, 0)[0] for x in tensor_unpadded] #list of length batch_size, each element is a tensor of shape (hid_dim*2,)
            # final = torch.stack(max)

            mask = lstm_out != 0
            masked_out = lstm_out.masked_fill(~mask, float('-inf'))
            final = torch.max(masked_out, dim=1)[0]
           
        else:
            concat_hidden_dir = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
            #unsort by length
            final = concat_hidden_dir.index_select(0, Variable(idx_unsort.cuda() if torch.cuda.is_available() else idx_unsort))
            
        return final
    

