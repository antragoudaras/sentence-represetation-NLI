import torch.nn as nn 
import torch

class Clasiifier(nn.Module):
    """Classifier of the model, along with concatinations"""

    def __init__(self, input_dim, hidden_dim = 512):
        super(Clasiifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = 3

        self.MLP = nn.Sequential(
            nn.Linear(4*self.input_dim, self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(self, encoded_premise, encoded_hypothesis):
        """Concat the inputs of the encoders and pass through the MLP"""
        element_wise_mult = torch.mul(encoded_premise, encoded_hypothesis)

        abs_diff = torch.abs(torch.sub(encoded_premise, encoded_hypothesis))

        final = torch.cat((encoded_premise, encoded_hypothesis, abs_diff, element_wise_mult), 1)

        return self.MLP(final)