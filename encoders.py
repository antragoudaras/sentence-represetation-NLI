import torch.nn as nn
import torch 
from torch.nn.utils.rnn import pack_padded_sequence

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
    pass