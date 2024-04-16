import torch.nn as nn

class Model(nn.Module):
    """Model for sentence classification."""
    def __init__(self, encoder, classifier):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, premise_tensor, premise_length_tensor, hypothesis_tensor, hypothesis_length_tensor):
        premise_rep = self.encoder(premise_tensor, premise_length_tensor)
        hypothesis_rep = self.encoder(hypothesis_tensor, hypothesis_length_tensor)
        out = self.classifier(premise_rep, hypothesis_rep)
        return out