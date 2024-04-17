import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=2)

# Define parameters
N, L, H_in, H_out, num_layers = 10, 4, 6, 3, 1
# Create random batch
batch = torch.rand(N, L, H_in)
# Create GRU layer (IMPORTANT: batch_first=True in this example!)
lstm = nn.LSTM(input_size=H_in, hidden_size=H_out, num_layers=num_layers, bidirectional=True, batch_first=True)
# Push batch through GRU layer
output, (h_n, _) = lstm(batch)
output = output.view(N, L, 2, H_out)
print('kippo')