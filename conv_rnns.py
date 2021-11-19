import typing
import torch
import torch.nn as nn
from torch.functional import Tensor

class ConvRNN(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
        self.intermediate_channels = intermediate_channels

    def forward(self, x:Tensor, state:Tensor) -> Tensor:
        x = torch.cat([x, state], dim=1)
        x = self.conv_x(x)
        return torch.tanh(x)

    def init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> Tensor:
        width, height = spatial_dim
        y = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return y


class ConvLSTMCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super(ConvLSTMCell, self).__init__()
        """conv_x  has a valid padding by:
        setting padding = kernel_size // 2
        hidden channels for h & c = intermediate_channels
        """
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple:
        """
        c and h channels = intermediate_channels so  a * c is valid
        if the last dim in c not equal to a then a has been halved
        """
        c, h = state
        h = h.to(device=x.device)
        c = c.to(device=x.device)
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a * c +  g * d
        h = b * torch.tanh(c)
        return c, h

    def init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> typing.Tuple:
        width, height = spatial_dim
        c = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return c, h

class ConvGRUCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3, stride=None):
        super(ConvGRUCell, self).__init__()
        """
        x dim = 3
        b * h = b & h should have same dim
        hidden channels for h = intermediate_channels
        """
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels * 2,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
        self.conv_y = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,  bias=True
        )
        self.intermediate_channels = intermediate_channels
    def forward(self, x:Tensor, h:Tensor) -> Tensor:
        y = x.clone()
        h = h.to(device=x.device)
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        a, b = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        y = torch.cat([y, b * h], dim=1)
        y = torch.tanh(self.conv_y(y))
        h = a * h + (1 - a) * y
        return h

    def _init_state(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> Tensor:
        width, height = spatial_dim
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return h

class ConvLSTM(nn.Module):
    def __init__(self, num_cells:int, in_channels:int, intermediate_channels:int, kernel_size:int, return_sequence:bool=False, return_cell_states:bool=False):
        super().__init__()
        self.num_cells = num_cells
        self.kernel_size = kernel_size
        self.intermediate_channels = intermediate_channels
        self.return_sequence = return_sequence
        self.return_cell_states = return_cell_states
        lstm_cells = []
        for _ in range(num_cells):
            lstm_cells.append(ConvLSTMCell(in_channels, intermediate_channels, kernel_size))
        self.lstm_cells = nn.ModuleList(lstm_cells)

    def forward(self, x:Tensor, state:typing.Tuple=None) -> typing.Tuple:
        b, s, _, w, h = x.shape
        cell_states = []

        for n in range(self.num_cells):
            seq_states = []
            for i in range(s):
                if i == 0 and n == 0: #first cell, first sequence
                    prev_state = self.lstm_cells[n].init_states(b, self.intermediate_channels, (w, h))
                elif  n > 0 and i > 0: #not first cell but sequence > 0
                    prev_state = seq_states[n - 1]
                elif n > 0 and i == 0: #not first cell but starting first sequence
                    prev_state = cell_states[i - 1][-1]
                else:
                    NotImplementedError()
                current_state = self.lstm_cells[n](x[:, i, :, :, :], prev_state)
                seq_states.append(current_state)
            cell_states.append(seq_states)

        cell_seq_states = cell_states[:][-1]
        last_cell_state = cell_seq_states[-1]

        if self.return_cell_states and self.return_sequence:
            return cell_seq_states, cell_states
        elif not self.return_sequence and not self.return_cell_states:
            return last_cell_state, None

        return last_cell_state , cell_states


class ConvGRU(nn.Module):
    def __init__(self, num_cells:int, in_channels:int, intermediate_channels:int, kernel_size:int, return_sequence:bool=False, return_cell_states:bool=False):
        super().__init__()
        self.num_cells = num_cells
        self.kernel_size = kernel_size
        self.intermediate_channels = intermediate_channels
        self.return_sequence = return_sequence
        self.return_cell_states = return_cell_states
        gru_cells = []
        for _ in range(num_cells):
            gru_cells.append(ConvLSTMCell(in_channels, intermediate_channels, kernel_size))
        self.gru_cells = nn.ModuleList(gru_cells)

    def forward(self, x:Tensor, state:typing.Tuple=None) -> typing.Tuple:
        b, s, _, w, h = x.shape
        cell_states = []

        for n in range(self.num_cells):
            seq_states = []
            for i in range(s):
                if i == 0 and n == 0: #first cell, first sequence
                    prev_state = self.gru_cells[n].init_states(b, self.intermediate_channels, (w, h))
                elif  n > 0 and i > 0: #not first cell but sequence > 0
                    prev_state = seq_states[n - 1]
                elif n > 0 and i == 0: #not first cell but starting first sequence
                    prev_state = cell_states[i - 1][-1]
                else:
                    NotImplementedError()
                current_state = self.gru_cells[n](x[:, i, :, :, :], prev_state)
                seq_states.append(current_state)
            cell_states.append(seq_states)

        cell_seq_states = cell_states[:][-1]
        last_cell_state = cell_seq_states[-1]

        if self.return_cell_states and self.return_sequence:
            return cell_seq_states, cell_states
        elif not self.return_sequence and not self.return_cell_states:
            return last_cell_state, None

        return last_cell_state , cell_states




if __name__ == "__main__":
    x = torch.randn((4, 10 , 3, 24, 24))
    x = x[:, 1, :, :, : ]
    h= torch.zeros((4, 128, 24, 24))
    s = ConvLSTMCell(3, 128, 3, stride=2)
    y = s(x, (h, h))
    print(y[0].shape)



