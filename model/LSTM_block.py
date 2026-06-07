import torch
import torch.nn as nn
import math

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Optimization of the GPU usage by combining the gates weights in 1 MLP ! [Standard]
        # Gates: Forget (f), Input (i), Cell Candidate (c), Output (o)
        self.gates_layer = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weights using standard Xavier/Glorot hidden state initialization
        # Ensure the right scaling of the vectors f, i, c_candidate and o [Only the x_state input are non-zero]
        # Hidden state at the start are equals to 0 ! Weights need to be initiated based only on the x_state length = hidden_state length
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv) # Ensure a variance of 1 during the sumation of the weights with the inputs value !

    def forward(self, x, h_prev, c_prev):
        """
        x:      [Batch, Input_Size] (Current time step features)
        h_prev: [Batch, Hidden_Size] (Short-term memory from last step)
        c_prev: [Batch, Hidden_Size] (Long-term memory from last step)
        """
        #Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1) # [Batch, Input_Size + Hidden_Size]
        
        # Project for all gates at once
        gates = self.gates_layer(combined) # [Batch, 4*Hidden_Size]
        
        # Split the tensor into the 4 original gates
        f_gate, i_gate, c_candidate, o_gate = torch.chunk(gates, chunks=4, dim=1)
        
        # Activation fun
        f_out = torch.sigmoid(f_gate)       # Forget gate
        i_out = torch.sigmoid(i_gate)       # Input gate
        cell_out = torch.tanh(c_candidate)     # Candidate state cell update
        o_out = torch.sigmoid(o_gate)       # Output gate
        
        # LT cell and ST state
        cell_next = (f_out * c_prev) + (i_out * cell_out) # [Batch, Hidden_Size]
        h_next = o_out * torch.tanh(cell_next) # [Batch, Hidden_Size]
        
        return h_next, cell_next
    

class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Instantiate our single-step cell engine
        self.cell = CustomLSTMCell(input_size, hidden_size)
        
    def forward(self, x):
        """
        x: [Batch, Time_Steps, Input_Size] (e.g., [Batch, 4, 128])
        """
        batch_size, time_steps, _ = x.shape
        device = x.device
        
        # 1. Initialize hidden state (h) and cell state (c) to zeros for step 0
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Array to store outputs for every step if needed
        outputs = []
        
        # Loop through the timeframe ! [4 Frames]
        for t in range(time_steps):
            x_t = x[:, t, :] # Extract step t -> Shape: [Batch, Input_Size]
            
            # Feed current step + past memory into our custom cell equations
            h, c = self.cell(x_t, h, c)
            
            outputs.append(h.unsqueeze(1))
            
        # Stack all sequential outputs back together
        outputs = torch.cat(outputs, dim=1) # Shape: [Batch, TimeFrames, Hidden_Size]
        
        # Return the hidden_state as the standrad LSTM library ! [Num_layer, Batch, Features]
        return outputs, (h.unsqueeze(0), c.unsqueeze(0))