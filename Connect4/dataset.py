import torch
import numpy as np

def convert_board(board):
    # Convert 6x7 board to 2x6x7 tensor
    player1_layer = (board == 1).astype(float)
    player2_layer = (board == 2).astype(float)
    stacked_arrays = np.stack([player1_layer, player2_layer])
    return torch.tensor(stacked_arrays, dtype=torch.float32)



class Connect4Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        state_t = convert_board(sample['state_t'])
        action = sample['action']
        state_t_plus_1 = convert_board(sample['state_t_plus_1'])
        outcome = sample['outcome']
        return state_t, action, state_t_plus_1, outcome
