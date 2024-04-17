import torch.nn as nn
import torch 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch.autograd import Variable

class BaselineEnc(nn.Module):
    """Average word embeddings to obtain sentence representations."""
    def __init__(self, glove_embeddings):
        super(BaselineEnc, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embeddings)
        self.embeddings.requires_grad = False

    def forward(self, indices, lengths):
        embeddings = self.embeddings(indices)
        #sum alonmg the sequence dimension
        sum_embeddings = embeddings.sum(dim=1).view(-1,1).to(torch.float32)
        #average embeedings
        return sum_embeddings / lengths

class UniLSTM(nn.Module): 
    """Uni-directional LSTM encoder."""

    def __init__(self,  glove_embedding, input_dim=300, hid_dim=2048):
        super(UniLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embedding)
        self.embeddings.requires_grad = False

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True)

    def forward(self, indices, lengths):

        embdgs = self.embeddings(indices)
        length = lengths.to('cpu')

        packed_embdgs = pack_padded_sequence(embdgs, length, batch_first=True, enforce_sorted=False)

        (hidden_state, _) = self.lstm(packed_embdgs)[1]

        #return the last hidden state as the sentence representation
        return hidden_state[-1] #last batch element
    
class BiLSTM(nn.Module):
    """Bi-directional LSTM encoder."""

    def __init__(self, glove_embedding, input_dim=300, hid_dim=2048, max_pooling=False):
        super(BiLSTM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(glove_embedding)
        self.embeddings.requires_grad = False

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True, bidirectional=True)
        self.max_pooling = max_pooling  # if True, use max pooling instead of concatenating the two hidden states

    def forward(self, indices, lengths):
        orig_emdgs = self.embeddings(indices).transpose(0, 1) # (seq_len, batch, input_size)

        #sort by lengts
        idx_sort = np.argsort(-lengths)
        idx_unsort = np.argsort(idx_sort)
        sent_len_sorted = -np.sort(-lengths)

        emdgs = orig_emdgs.index_select(1, Variable(idx_sort))
        emdgs2 = orig_emdgs.index_select(1, idx_sort)
        
        # Sort by lengths
        # sorted_lengths2, sorted_indices2 = torch.sort(lengths, descending=True)
        # _, unsorted_indices2 = torch.sort(sorted_indices2)

        # emdgs2 = orig_emdgs.index_select(1, sorted_indices2)

        # lengths = lengths.to('cpu')
        sent_len_sorted = torch.tensor(sent_len_sorted).to('cpu')

        # packed_emdgs = pack_padded_sequence(emdgs, lengths, batch_first=True, enforce_sorted=False)

        packed_emdgs = pack_padded_sequence(emdgs, sent_len_sorted)

        packed_output, (hidden_states, _) = self.lstm(packed_emdgs)

        # lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        lstm_out = pad_packed_sequence(packed_output, batch_first=True)[0]

        #unsort by length 
        lstm_out1 = lstm_out.index_select(0, Variable(idx_unsort))
        lstm_out2 = lstm_out.index_select(0, idx_unsort)


        # seq_len = emdgs.size(1)
        # batch_size = emdgs.size(0)
        # num_directions = 2

        # output = lstm_out.view(batch_size, seq_len, num_directions, self.lstm.hidden_size)        
        #get the forward state from output and hidden state and check they are the same

        return hidden_states[-1] if self.max_pooling else torch.cat((hidden_states[0], hidden_states[-1]), dim=1)
    

