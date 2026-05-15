"""JEPA implementation optimized for Connect4 Learning"""

import copy, math, random, pickle
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import dataset
import board

def lr_warmup_cosine(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm: return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@torch.no_grad()
def ema_update(tgt, online, m):
    for pt, po in zip(tgt.parameters(), online.parameters()): 
        pt.mul_(m).add_(po.detach(), alpha=1 - m)


class Encoder(nn.Module):
    def __init__(self, in_chans=2, latent_dim=128):
        super().__init__()
        self.dim = latent_dim

        # Strided convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Compresses board size down to 3x4
            nn.ReLU()
        )
        self.flattened_size = 64 * 3 * 4
        self.proj = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, grids):
        B = grids.size(0)
        x = self.conv(grids)
        x = x.view(B, -1)
        return self.proj(x)


class Predictor(nn.Module):
    def __init__(self, enc_dim=128, action_dim=7, hidden_dim=512):
        super().__init__()
        input_dim = enc_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim)
        )

    def forward(self, z_t, action_one_hot):
        query = torch.cat([z_t, action_one_hot], dim=1)
        return self.net(query)


class ValueHead(nn.Module):
    def __init__(self, enc_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1), 
            nn.Tanh(), # To have value in [-1, +1]
        )

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1) # Bound coordinates explicitly before running evaluation
        return self.net(z).squeeze(-1)


def train_value_head(encoder, device, loader, epochs=8, lr=3e-3, wd=1e-4):
    encoder.eval()
    for param in encoder.parameters(): param.requires_grad_(False)
    value_head = ValueHead(enc_dim=encoder.dim).to(device)
    opt = torch.optim.AdamW(value_head.parameters(), lr=lr, weight_decay=wd)
    print("\nTraining Value Head Head on Structured Vectors...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for state_t, action, state_t_plus_1, outcome in loader:
            state_t = state_t.to(device)
            outcome = outcome.to(device).float()

            with torch.no_grad():
                z_t = encoder(state_t)

            pred_value = value_head(z_t)
            loss = F.smooth_l1_loss(pred_value, outcome)

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Value Head Training - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(loader):.4f}")
    return value_head


def train(epochs=10, batch_size=256, lr=3e-4, wd=0.05, ema_start=0.996, ema_end=1.0, device=None):
    device = device or pick_device(); print(f"device: {device}")

    with open('data/connect4_self_play_heuristic_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    ds = dataset.Connect4Dataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    ctx_enc = Encoder().to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)

    pred = Predictor().to(device)
    inverse_head = nn.Linear(ctx_enc.dim * 2, 7).to(device)

    opt = torch.optim.AdamW(
        list(ctx_enc.parameters()) + list(pred.parameters()) + list(inverse_head.parameters()), 
        lr=lr, weight_decay=wd
    )

    total = epochs * len(loader); step = 0

    for epoch in range(epochs):
        for state_t, action, state_t_plus_1, outcome in loader:
            state_t = state_t.to(device)           
            a_t = action.to(device)            
            z_t_plus_1 = state_t_plus_1.to(device)  
            a_t_one_hot = F.one_hot(a_t, num_classes=7).float()

            for g in opt.param_groups: g["lr"] = lr_warmup_cosine(step, total, lr)

            # Latent Maps
            ce = ctx_enc(state_t)
            with torch.no_grad(): 
                full = tgt_enc(z_t_plus_1)

            ce_norm = F.normalize(ce, p=2, dim=-1)
            full_norm = F.normalize(full, p=2, dim=-1)

            # Dynamic Forward Executions
            pred_z_t_plus_1 = pred(ce_norm, a_t_one_hot)
            pred_z_t_plus_1_norm = F.normalize(pred_z_t_plus_1, p=2, dim=-1)

            state_transition = torch.cat([ce_norm, full_norm], dim=1)
            pred_action_logits = inverse_head(state_transition)

            # Joint Loss Balancing
            matching_loss = F.smooth_l1_loss(pred_z_t_plus_1_norm, full_norm)
            inv_loss = F.cross_entropy(pred_action_logits, a_t)
            loss = matching_loss + inv_loss * 0.5

            opt.zero_grad(); loss.backward(); opt.step()

            m = ema_start + (ema_end - ema_start) * (step / max(1, total - 1))
            ema_update(tgt_enc, ctx_enc, m)

            if step % 100 == 0:
                print(f"ep={epoch} step={step:5d} loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")
            step += 1

    value_head = train_value_head(ctx_enc, device, loader)
    return {"ctx_enc": ctx_enc, "predictor": pred, "value_head": value_head, "device": device}


def jepa_choose_move(board_matrix, active_player, encoder, predictor, value_head, device):
    encoder.eval()
    predictor.eval()
    value_head.eval()

    # Channel 0 is ALWAYS Player 1, Channel 1 is ALWAYS Player 2
    channel_1 = (board_matrix == 1).astype(np.float32)
    channel_2 = (board_matrix == 2).astype(np.float32)
    state_tensor = torch.tensor(np.stack([channel_1, channel_2])).unsqueeze(0).to(device) 

    with torch.no_grad():
        z_t = encoder(state_tensor)
        z_t_norm = F.normalize(z_t, p=2, dim=-1) 

    valid_moves = board.get_valid_locations(board_matrix)
    best_action = random.choice(valid_moves) if valid_moves else -1

    # ALWAYS MAXIMIZE: +1.0 in the dataset means the move was used by a winning player
    highest_score = -float('inf') 

    for action in valid_moves:
        action_tensor = torch.tensor([action]).to(device)
        action_one_hot = F.one_hot(action_tensor, num_classes=7).float()

        with torch.no_grad():
            imagined_z_next = predictor(z_t_norm, action_one_hot)
            imagined_z_next_norm = F.normalize(imagined_z_next, p=2, dim=-1)
            score = value_head(imagined_z_next_norm).item()

        print(f"  -> Imagining column {action} | Predicted Board Value: {score:+.4f}")

        if score > highest_score:
            highest_score = score
            best_action = action

    return best_action


def play_vs_jepa(trained_modules):
    ctx_enc = trained_modules["ctx_enc"]
    pred = trained_modules["predictor"]
    value_head = trained_modules["value_head"]
    device = trained_modules["device"]

    b = board.create_board()
    turn = 0

    print("\n=== GAME START: HUMAN vs NEURAL WORLD MODEL ===")
    board.print_board(b)

    while not board.game_over(b):
        current_player = 1 if turn == 0 else 2

        if current_player == 1:
            valid_moves = board.get_valid_locations(b)
            while True:
                try:
                    col = int(input(f"Your Turn! Choose column {valid_moves}: "))
                    if col in valid_moves:
                        break
                    print("Invalid column choice.")
                except ValueError:
                    print("Please enter a valid column integer.")
        else:
            print("\nJEPA is thinking (Running latent state projections)...")
            col = jepa_choose_move(b, current_player, ctx_enc, pred, value_head, device)
            print(f"JEPA plays column: {col}")

        row = board.get_next_open_row(b, col)
        board.drop_piece(b, row, col, current_player)
        board.print_board(b)

        if board.winning_move(b, current_player):
            winner_text = "You win" if current_player == 1 else "JEPA Wins!"
            print(f"GAME OVER: {winner_text}")
            return

        if len(board.get_valid_locations(b)) == 0:
            print("GAME OVER: It's a draw")
            return

        turn = 1 - turn

if __name__ == "__main__":
    trained_components = train()
    play_vs_jepa(trained_components)
