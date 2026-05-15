import numpy as np
import torch
import board
import pickle


def run_self_play_game():
    b = board.create_board()
    game_history = []
    turn = 0  # 0 for Player 1, 1 for Player 2

    while not board.game_over(b):
        current_player = 1 if turn == 0 else 2
        valid_moves = board.get_valid_locations(b)

        # 1. Choose a random valid action
        action = np.random.choice(valid_moves)

        # 2. Save the state before the move
        state_t = np.copy(b)

        # 3. Execute the move in the game environment
        row = board.get_next_open_row(b, action)
        board.drop_piece(b, row, action, current_player)

        # 4. Record transition (State_t, Action, Player_Who_Moved)
        game_history.append({
            'state_t': state_t,
            'action': action,
            'player': current_player,
            'state_t_plus_1': np.copy(b)
        })

        turn = 1 - turn  # Switch turn

    # GAME OVER PHASE
    # 5. Determine ultimate game outcome from the engine
    if board.winning_move(b, 1):
        winner = 1
    elif board.winning_move(b, 2):
        winner = 2
    else:
        winner = 0  # Tie

    # 6. Post-process the history to assign supervised values
    processed_dataset = []
    for step in game_history:
        # Determine value label from perspective of the active player
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
    # Simulate multiple self-play games to build a dataset
    number_of_games = 5000
    full_dataset = []
    for _ in range(number_of_games):
        game_data = run_self_play_game()
        full_dataset.extend(game_data)
        if (_ + 1) % 100 == 0:
            print(f"Completed {_ + 1} games...")
    print(f"Generated dataset with {len(full_dataset)} samples from {number_of_games} games.")

    # TODO: See if we could convert the dataset to Torch now, or later

    # Store the dataset in Pickle format for later use
    with open('connect4_self_play_dataset.pkl', 'wb') as f:
        pickle.dump(full_dataset, f)


if __name__ == "__main__":
    main()
