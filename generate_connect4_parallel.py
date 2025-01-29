################################################################################
########## Imports. Some are not needed anymore but remain for now #############
################################################################################
import pickle
import random
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from IPython.display import clear_output
from termcolor import colored
from tqdm import tqdm

# https://www.youtube.com/watch?v=UXW2yZndl7U

################################################################################
####### Define the bitboard class, add piece, and check win ####################
################################################################################

class Bitboard:
    def __init__(self):
        self.player1 = 0  # Bitboard for Player 1
        self.player2 = 0  # Bitboard for Player 2
        self.height = [0] * 7  # Tracks the next free row in each column
        self.rows = 6
        self.cols = 7

    def print_board(self):
        """Prints the board in a human-readable format."""
        board = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        for col in range(self.cols):
            for row in range(self.rows):
                index = col * self.rows + row
                if self.player1 & (1 << index):
                    board[row][col] = colored('X', 'red') # Player 1
                elif self.player2 & (1 << index):
                    board[row][col] = colored('O', 'yellow')  # Player 2
        for row in reversed(board):  # Print top row first
            print('|' + '|'.join(row) + '|')
        print('-' * (2 * self.cols + 1))


# replaces the update_board fcn
def add_piece(bitboard, column, player):
    """
    Add a piece to the given column for the specified player.
    :param bitboard: Bitboard instance.
    :param column: The column (0-6) to play in.
    :param player: 1 (Player 1) or 2 (Player 2).
    """
    if column < 0 or column >= bitboard.cols or bitboard.height[column] >= bitboard.rows:
        raise ValueError("Invalid column or column is full.")
    index = column * bitboard.rows + bitboard.height[column]
    if player == 1:
        bitboard.player1 |= (1 << index)  # Set bit in Player 1's bitboard
    else:
        bitboard.player2 |= (1 << index)  # Set bit in Player 2's bitboard
    bitboard.height[column] += 1  # Update the next free row in the column

# replaces the check_for_win fcn
def check_winner(bitboard, player):
    """
    Check if the given player has won.
    :param bitboard: Bitboard instance.
    :param player: 1 (Player 1) or 2 (Player 2).
    :return: True if the player has won, False otherwise.
    """
    board = bitboard.player1 if player == 1 else bitboard.player2
    rows = bitboard.rows
    cols = bitboard.cols

    # Check horizontal
    horizontal = board & (board >> 1)
    if horizontal & (horizontal >> 2):
        return True

    # Check vertical
    vertical = board & (board >> rows)
    if vertical & (vertical >> (2 * rows)):
        return True

    # Check diagonal (top-left to bottom-right)
    diagonal1 = board & (board >> (rows + 1))
    if diagonal1 & (diagonal1 >> (2 * (rows + 1))):
        return True

    # Check anti-diagonal (top-right to bottom-left)
    diagonal2 = board & (board >> (rows - 1))
    if diagonal2 & (diagonal2 >> (2 * (rows - 1))):
        return True

    return False




################################################################################
############### MCTS function and its helper functions #########################
################################################################################

# updates find_legal fcn
def find_legal(bitboard):
    """
    Find all legal moves (columns not full).
    :param bitboard: Bitboard instance.
    :return: List of legal column indices.
    """
    return [col for col in range(bitboard.cols) if bitboard.height[col] < bitboard.rows]

def look_for_win(bitboard, player):
    """
    Check if the current player has an immediate winning move.
    :param bitboard: The current board state (Bitboard instance).
    :param player: The current player (1 = Player 1, 2 = Player 2).
    :return: The column index of a winning move, or -1 if no such move exists.
    """
    legal_moves = find_legal(bitboard)  # Get all legal columns
    for col in legal_moves:
        # Simulate placing a piece in the column
        temp_board = Bitboard()
        temp_board.player1, temp_board.player2 = bitboard.player1, bitboard.player2
        temp_board.height = bitboard.height[:]
        add_piece(temp_board, col, player)

        # Check if this move results in a win
        if check_winner(temp_board, player):
            return col  # Found a winning move
    return -1  # No winning move found

def find_all_nonlosers(bitboard, player):
    """
    Find all legal moves that do not allow the opponent to win immediately.
    :param bitboard: The current board state (Bitboard instance).
    :param player: The current player (1 = Player 1, 2 = Player 2).
    :return: A list of column indices for "safe" moves.
    """
    opponent = 3 - player  # Opponent player (1 ↔ 2)
    legal_moves = find_legal(bitboard)  # Get all legal columns
    safe_moves = []

    for col in legal_moves:
        # Simulate the current player's move
        temp_board = Bitboard()
        temp_board.player1, temp_board.player2 = bitboard.player1, bitboard.player2
        temp_board.height = bitboard.height[:]
        add_piece(temp_board, col, player)

        # Check if the opponent has an immediate winning move after this
        opponent_wins = False
        for opp_col in find_legal(temp_board):
            # Simulate the opponent's move
            temp_temp_board = Bitboard()
            temp_temp_board.player1, temp_temp_board.player2 = temp_board.player1, temp_board.player2
            temp_temp_board.height = temp_board.height[:]
            add_piece(temp_temp_board, opp_col, opponent)

            if check_winner(temp_temp_board, opponent):
                opponent_wins = True
                break

        # If no immediate win for the opponent, add the move to safe moves
        if not opponent_wins:
            safe_moves.append(col)

    return safe_moves

# updates rollout fcn
def rollout(bitboard, next_player):
    """
    Perform a random rollout from the given board state.
    :param bitboard: Bitboard instance.
    :param next_player: 1 (Player 1) or 2 (Player 2).
    :return: The winner ('X', 'O', or 'tie').
    """
    current_player = next_player
    while True:
        legal = find_legal(bitboard)
        if not legal:
            return 'tie'  # No legal moves = tie
        move = random.choice(legal)
        add_piece(bitboard, move, current_player)
        if check_winner(bitboard, current_player):
            return 'X' if current_player == 1 else 'O'
        current_player = 3 - current_player  # Switch between 1 and 2

def mcts(bitboard, color0, nsteps):
    """
    Perform Monte Carlo Tree Search (MCTS) to determine the best move.
    :param bitboard: Bitboard instance representing the current board state.
    :param color0: Starting player ('X' = Player 1, 'O' = Player 2).
    :param nsteps: Number of MCTS simulations to run.
    :return: The best column to play.
    """
    # Initialize the MCTS dictionary with the current board state.
    # Key: (player1_bitboard, player2_bitboard)
    # val: [total visits, total score]
    mcts_dict = {(bitboard.player1, bitboard.player2): [0, 0]}
    # Step 1: Immediate checks
    # get all legal moves
    legal_moves = find_legal(bitboard)

    # check for an immediate winning move
    win_move = look_for_win(bitboard, 1 if color0 == 'X' else 2)
    if win_move != -1:
        return win_move  # Return the winning move immediately

    # check for moves that let the opponent win immediately
    safe_moves = find_all_nonlosers(bitboard, 1 if color0 == 'X' else 2)
    if safe_moves:  # If there are safe moves, use them
        legal_moves = safe_moves
    # else:
    #     print("Warning: No safe moves found; using all legal moves.")

    # If no safe moves exist, move forward with all legal moves

    # Step 2: Initialize MCTS simulation
    for _ in range(nsteps):  # Perform `nsteps` simulations
        path = []  # Path of visited states
        # Copy the initial board
        temp_board = Bitboard()
        temp_board.player1, temp_board.player2 = bitboard.player1, bitboard.player2
        temp_board.height = bitboard.height[:]
        current_player = 1 if color0 == 'X' else 2 # current player (1 = Player 1, 2 = Player 2)
        winner = None

        # Selection & Expansion Phase
        while True:
            state_key = (temp_board.player1, temp_board.player2) # Represent the current board state as a tuple
            path.append(state_key) # Add this state to the path

            # If the state is not in the MCTS dictionary, initialize it
            if state_key not in mcts_dict:
                mcts_dict[state_key] = [0, 0]  # [total visits, total score]
                win_move = look_for_win(temp_board, current_player)
                if win_move != -1:
                    add_piece(temp_board, win_move, current_player)
                    winner = 'X' if current_player == 1 else 'O'
                    break
                break

            # UCB1 calculation to select the best move
            legal_moves = find_legal(temp_board)
            safe_moves = find_all_nonlosers(temp_board, current_player)
            if safe_moves:
                legal_moves = safe_moves
            if not legal_moves:
                winner = 'tie'  # No moves = tie
                break

            ucb1_scores = []
            parent_visits = mcts_dict[state_key][0]
            for col in legal_moves:
                # Generate next state
                next_board = Bitboard()
                next_board.player1, next_board.player2 = temp_board.player1, temp_board.player2
                next_board.height = temp_board.height[:]
                add_piece(next_board, col, current_player)
                next_key = (next_board.player1, next_board.player2)

                # UCB1 calculation
                if next_key not in mcts_dict: # if total visits == 0
                    ucb1_scores.append(float('inf'))  # Prioritize unexplored states
                else:
                    visits, score = mcts_dict[next_key]
                    # exploitation term + exploration term
                    # exploitation term: how favorable this move has been so far
                    # exploration term: encourages exploration of moves visited less oftern relative to parent state
                    ucb1 = (score / visits) + 2 * np.sqrt(np.log(parent_visits) / (1 + visits))
                    ucb1_scores.append(ucb1)

            # Choose the best move based on UCB1
            # will choose the highest ucb1 score
            # unvisited states will be inf and will be chosen
            best_move_idx = np.argmax(ucb1_scores)
            chosen_move = legal_moves[best_move_idx]
            add_piece(temp_board, chosen_move, current_player)

            # check if the move wins the game
            if check_winner(temp_board, current_player):
                winner = 'X' if current_player == 1 else 'O'
                break

            # game is not over so switch to the other player
            current_player = 3 - current_player # switch between Player 1 and Player 2

        # Rollout Phase
        # if the while loop is exited due to new state discovery
        if winner is None:
            winner = rollout(temp_board, current_player)

        # Backpropagation Phase
        for i, state in enumerate(reversed(path)):
            mcts_dict[state][0] += 1  # Increment visits
            if winner == 'tie':
                continue  # No score adjustment for ties
            if (winner == 'X' and (i % 2 == 0)) or (winner == 'O' and (i % 2 == 1)):
                mcts_dict[state][1] += 1  # Favorable for the player who made the move
            else:
                mcts_dict[state][1] -= 1  # Unfavorable for the player who made the move

    # Step 3: Choose the Best Move
    best_score = -float('inf')
    best_col = -1
    for col in find_legal(bitboard):
        temp_board = Bitboard()
        temp_board.player1, temp_board.player2 = bitboard.player1, bitboard.player2
        temp_board.height = bitboard.height[:]
        add_piece(temp_board, col, 1 if color0 == 'X' else 2)
        state_key = (temp_board.player1, temp_board.player2)
        if state_key in mcts_dict:
            visits, score = mcts_dict[state_key]
            avg_score = score / visits if visits > 0 else -float('inf')
            if avg_score > best_score:
                best_score = avg_score
                best_col = col

    return best_col



################################################################################
#################### Functions for Dataset Creation ############################
################################################################################

def save_dataset(dataset, save_path):
    """
    Save the dataset to a file.
    :param dataset: The dataset to save.
    :param save_path: Path to save the dataset.
    """
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def generate_single_game(nsteps):
    """
    Simulate a single Connect 4 game and update the progress counter.
    :param nsteps: Number of MCTS simulations per move.
    :return: A list of states and best moves for the game.
    """
    dataset = []  # Store states for this game
    bitboard = Bitboard()  # Initialize an empty board
    current_player = 'X'

    # Simulate the game
    while True:
        if check_winner(bitboard, 1):
            break  # Player 1 wins
        if check_winner(bitboard, 2):
            break  # Player 2 wins
        if not find_legal(bitboard):  # No legal moves = tie
            break
        # Use MCTS to determine the best move
        skill = random.randint(500, nsteps)
        best_move = mcts(bitboard, current_player, skill)
        # Record the state and best move
        if skill >= 2000:
            dataset.append({
                'state': (bitboard.player1, bitboard.player2),
                'best_move': best_move
            })
        # Apply the move
        add_piece(bitboard, best_move, 1 if current_player == 'X' else 2)
        # Switch players
        current_player = 'O' if current_player == 'X' else 'X'
    return dataset

def worker_single_game(game_id, nsteps):
    """
    A top-level function that calls single-game simulation.
    Must be defined at the module level so it can be pickled on Windows.
    """
    return generate_single_game(nsteps)
    # print("Worker started")
    # result = generate_single_game(nsteps)
    # print("Worker finished")
    # return result

def parallel_generate_dataset(num_games, nsteps, save_path, processes):
    """
    Parallel dataset generation using a top level worker fcn with progress tracking using tqdm.
    :param num_games: Number of games to simulate.
    :param nsteps: Number of MCTS simulations per move.
    :param save_path: Path to save the generated dataset.
    :param processes: Number of parallel processes to use.
    """
    worker = partial(worker_single_game, nsteps=nsteps)
    dataset = []

    # Initialize a multiprocessing pool
    with Pool(processes=processes) as pool:
        # print("pooling")
        # Use tqdm to track progress
        with tqdm(total=num_games, desc="Generating games") as pbar:
            # print("tqdming")
            # run_game = partial(generate_single_game, nsteps)
            for game_result in pool.imap_unordered(worker, range(num_games)):
                # print("extending")
                dataset.extend(game_result)
                pbar.update(1)

    # Save the dataset to a file
    save_dataset(dataset, save_path)
    print(f"Dataset generation complete! Saved {len(dataset)} states to {save_path}")




################################################################################
############### Executable code: Dataset creation in parallel ##################
################################################################################


if __name__ == "__main__":
    ############################################################################
    ############### The following 4 variables are adjustable ###################
    ############################################################################

    num_games = 8000  # Number of games to simulate
    nsteps = 4000      # Number of MCTS simulations per move
    save_path = "connect4_rand.pkl"  # Path to save the dataset
    processes = 12 # Number of parallel processes
    parallel_generate_dataset(
        num_games=num_games,  # Number of games to simulate
        nsteps=nsteps,    # Number of MCTS simulations per move
        save_path=save_path,  # Path to save the dataset
        processes=processes     # Number of parallel processes
    )


