import numpy as np
import pickle
import random
import board

# --- OPTIMIZED MINIMAX ENGINE WITH ALPHA-BETA PRUNING ---

def evaluate_window(window, player):
    """Assigns strategic scores to a local slice of 4 slots on the board."""
    score = 0
    opponent = 2 if player == 1 else 1

    p_count = np.count_nonzero(window == player)
    o_count = np.count_nonzero(window == opponent)
    empty_count = np.count_nonzero(window == 0)

    if p_count == 4:
        score += 1000
    elif p_count == 3 and empty_count == 1:
        score += 50  # Strong offensive layout
    elif p_count == 2 and empty_count == 2:
        score += 10  # Positional buildup

    if o_count == 3 and empty_count == 1:
        score -= 80  # Aggressive defensive block tracking
    
    return score

def score_position(b, player):
    """Evaluates non-terminal board configurations using sliding windows."""
    score = 0

    # 1. Prioritize Center Column Control (highly strategic zone)
    center_array = b[:, 3]
    center_count = np.count_nonzero(center_array == player)
    score += center_count * 15

    # 2. Horizontal Windows
    for r in range(6):
        row_array = b[r, :]
        for c in range(4):
            window = row_array[c:c+4]
            score += evaluate_window(window, player)

    # 3. Vertical Windows
    for c in range(7):
        col_array = b[:, c]
        for r in range(3):
            window = col_array[r:r+4]
            score += evaluate_window(window, player)

    # 4. Positive-Slope Diagonals
    for r in range(3):
        for c in range(4):
            window = np.array([b[r+i, c+i] for i in range(4)])
            score += evaluate_window(window, player)

    # 5. Negative-Slope Diagonals
    for r in range(3, 6):
        for c in range(4):
            window = np.array([b[r-i, c+i] for i in range(4)])
            score += evaluate_window(window, player)

    return score

def is_terminal_node(b):
    """Checks if the current layout represents an endgame scenario."""
    return board.winning_move(b, 1) or board.winning_move(b, 2) or len(board.get_valid_locations(b)) == 0

def minimax(b, depth, alpha, beta, maximizingPlayer, current_player):
    """Alpha-Beta Minimax with center-out move ordering to maximize pruning velocity."""
    valid_moves = board.get_valid_locations(b)
    is_terminal = is_terminal_node(b)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if board.winning_move(b, current_player):
                return (None, 100000 + depth)
            elif board.winning_move(b, 2 if current_player == 1 else 1):
                return (None, -100000 - depth)
            else:
                return (None, 0)
        else:
            return (None, score_position(b, current_player))

    # Move Ordering: Sort valid columns from center outwards to trigger fast alpha-beta pruning cuts
    valid_moves.sort(key=lambda col: abs(3 - col))

    if maximizingPlayer:
        value = -float('inf')
        best_col = random.choice(valid_moves) if valid_moves else None
        for col in valid_moves:
            temp_board = np.copy(b)
            row = board.get_next_open_row(temp_board, col)
            board.drop_piece(temp_board, row, col, current_player)
            
            _, new_score = minimax(temp_board, depth - 1, alpha, beta, False, current_player)
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:
        value = float('inf')
        best_col = random.choice(valid_moves) if valid_moves else None
        opponent = 2 if current_player == 1 else 1
        for col in valid_moves:
            temp_board = np.copy(b)
            row = board.get_next_open_row(temp_board, col)
            board.drop_piece(temp_board, row, col, opponent)
            
            _, new_score = minimax(temp_board, depth - 1, alpha, beta, True, current_player)
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value


# --- HYBRID DATA SIMULATION GENERATOR ---

def hybrid_self_play_game(depth=2, epsilon=0.1):
    """
    Generates high-level self-play configurations.
    Uses 1-step heuristic for fast evaluations, activating deep minimax search 
    strictly during mid-game windows to maximize compilation speed.
    """
    b = board.create_board()
    game_history = []
    turn = 0  # 0 for Player 1, 1 for Player 2
    move_count = 0
    
    while not board.game_over(b):
        current_player = 1 if turn == 0 else 2
        opponent_player = 2 if current_player == 1 else 1
        valid_moves = board.get_valid_locations(b)

        # 1. Exploration Factor (Epsilon-Greedy introduces error recovery states)
        if random.random() < epsilon:
            action = random.choice(valid_moves)
        else:
            action = None
            
            # 2. HEURISTIC FAST PASS: Check immediate 1-step winning opportunities
            for col in valid_moves:
                temp_board = np.copy(b)
                row = board.get_next_open_row(temp_board, col)
                board.drop_piece(temp_board, row, col, current_player)
                if board.winning_move(temp_board, current_player):
                    action = col
                    break
            
            # Check immediate 1-step blocking opportunities
            if action is None:
                for col in valid_moves:
                    temp_board = np.copy(b)
                    row = board.get_next_open_row(temp_board, col)
                    board.drop_piece(temp_board, row, col, opponent_player)
                    if board.winning_move(temp_board, opponent_player):
                        action = col
                        break
            
            # 3. DEEP STRATEGIC PASS: Activate Minimax lookahead strictly for critical mid-game turns
            if action is None:
                if 6 <= move_count <= 24:
                    action, _ = minimax(b, depth, -float('inf'), float('inf'), True, current_player)
                
                # Opening/Late fallback to fast random choice if outside mid-game window
                if action is None or action not in valid_moves:
                    action = random.choice(valid_moves)

        # Save historical transition logs
        state_t = np.copy(b)
        row = board.get_next_open_row(b, action)
        board.drop_piece(b, row, action, current_player)

        game_history.append({
            'state_t': state_t,
            'action': action,
            'player': current_player,
            'state_t_plus_1': np.copy(b)
        })

        turn = 1 - turn
        move_count += 1

    # End-game checking logic
    if board.winning_move(b, 1):
        winner = 1
    elif board.winning_move(b, 2):
        winner = 2
    else:
        winner = 0

    processed_dataset = []
    for step in game_history:
        if winner == 0:
            value_label = 0.0
        elif winner == step['player']:
            value_label = 1.0  # Active player won
        else:
            value_label = -1.0  # Active player lost

        processed_dataset.append({
            'state_t': step['state_t'],
            'action': step['action'],
            'state_t_plus_1': step['state_t_plus_1'],
            'outcome': value_label
        })
    return processed_dataset


def main():
    number_of_games = 50000
    full_dataset = []
    
    # Depth=2 provides excellent balance between deep strategic setups and generation velocity
    print("Starting generation loop via Hybrid Alpha-Beta Engines...")
    for i in range(number_of_games):
        game_data = hybrid_self_play_game(depth=2, epsilon=0.1)
        full_dataset.extend(game_data)
        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1}/{number_of_games} games... Dataset Array Size: {len(full_dataset)}")
            
    print(f"\nGeneration complete. Total states collected: {len(full_dataset)}")

    with open('data/connect4_self_play_heuristic_dataset.pkl', 'wb') as f:
        pickle.dump(full_dataset, f)
    print("Dataset securely stored in: data/connect4_self_play_heuristic_dataset.pkl")

if __name__ == "__main__":
    main()
