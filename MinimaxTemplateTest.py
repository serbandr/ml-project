import pyspiel
from absl import app
import time
import re
import numpy as np
from collections import Counter

minimax_counter = 0
trans_counter = 0

num_cols = 0
num_rows = 0

# ------------------- EVERYTHING BELOW HERE IS FROM THEIR EXAMPLE TO USE AS HELPER FUNCTIONS -----------------------


def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical': 1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell': 2}  # Who has won the cell
    return p.get(part, part)


def state2num(state):
    s = {'e': 0, 'empty': 0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)


def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)


def get_observation(obs_tensor, state, row, col, part):
    num_cells = (num_rows + 1) * (num_cols + 1)
    num_parts = 3  # (horizontal, vertical, cell)

    state = state2num(state)
    part = part2num(part)
    idx = part \
          + (row * (num_cols + 1) + col) * num_parts \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]


def get_observation_state(obs_tensor, row, col, part, as_str=True):
    is_state = None
    for state in range(3):
        if get_observation(obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    if as_str:
        is_state = num2state(is_state)
    return is_state

# ------------------- EVERYTHING ABOVE HERE IS FROM THEIR EXAMPLE TO USE AS HELPER FUNCTIONS -----------------------


def _get_owned_cells(state, maximizing_player_id):
    # So basically get the current state, use the pre-given functions to get which players own which cells
    # and return the amount of owned cells for each player as an array [x, y]

    player1_cells = 0
    player2_cells = 0

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            owned_cell = get_observation_state(state.observation_tensor(maximizing_player_id), i, j, 'c')
            if owned_cell == "player1":
                player1_cells += 1
            elif owned_cell == "player2":
                player2_cells += 1

    return [player1_cells, player2_cells]


def _symmetric_key(state):
    """Generate symmetric keys for the given state."""

    # Use the dbns!

    dbns = state.dbn_string()

    # Split it up in 2 parts, one for the horizontal lines and one for the vertical lines

    num_horiz_lines = num_cols * (1 + num_rows)
    formatted_dbn_string = [dbns[0:num_horiz_lines], dbns[num_horiz_lines:]]

    # Split each part up further in sub-parts (per row or per columns so to speak)

    horiz_string = formatted_dbn_string[0]
    split_array = [''.join(horiz_string[i:i + num_cols]) for i in range(0, len(horiz_string), num_cols)]
    formatted_dbn_string[0] = split_array

    vert_string = formatted_dbn_string[1]
    split_array = [''.join(vert_string[i:i + num_rows]) for i in range(0, len(vert_string), num_rows)]

    # Now reformat the vertical part, so it's top to bottom vertically, and then left to right
    split_array_formatted = []
    for i in range(len(split_array[0])):
        split_array_formatted.append(''.join([elem[i] for elem in split_array]))
    formatted_dbn_string[1] = split_array_formatted

    # In the end for a 4x5 dots and boxes you'd have an array of the form
    # [['11111', '11111', '11111', '11111', '11111', '11111'], ['111101', '111111', '111101', '111011', '111110']]

    keys = []
    # Normal key
    keys.append(''.join([''.join(inner) for inner in formatted_dbn_string]))
    # Horizontal symmetry
    horiz_lines = formatted_dbn_string[0][::-1]
    keys.append(''.join([''.join(inner) for inner in [horiz_lines, formatted_dbn_string[1]]]))
    # Vertical symmetry
    vert_lines = formatted_dbn_string[1][::-1]
    keys.append(''.join([''.join(inner) for inner in [formatted_dbn_string[0], vert_lines]]))
    # Both symmetries
    keys.append(''.join([''.join(inner) for inner in [horiz_lines, vert_lines]]))

    return keys


def _minimax(state, maximizing_player_id, alpha=float('-inf'), beta=float('inf'), transposition_table=None):
    """
    Implements a min-max algorithm with transposition tables.

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.
      alpha: The best value that the maximizing player can guarantee at this node or above.
      beta: The best value that the minimizing player can guarantee at this node or above.
      transposition_table: A dictionary to cache the values of previously evaluated states.

    Returns:
      The optimal value of the sub-game starting in state
    """

    global minimax_counter, trans_counter
    minimax_counter += 1

    # Check if state is terminal or if the player ID is valid
    if state.is_terminal():
        # Calculate the win margin (nr. of boxes the maximizing player has over the minimizing player)

        margin = (_get_owned_cells(state, maximizing_player_id)[maximizing_player_id] -
                  _get_owned_cells(state, maximizing_player_id)[abs(maximizing_player_id - 1)])

        return margin

    if transposition_table is None:
        transposition_table = {}

    player = state.current_player()

    keys = _symmetric_key(state)

    print(len(transposition_table))

    for key in keys:
        if key in transposition_table:
            if transposition_table[key][3] == player:
                trans_counter += 1
                return transposition_table[key][0]

    # Use Alpha-Beta pruning to speed up the process
    if player == maximizing_player_id:
        value = float('-inf')
        for action in state.legal_actions():
            value = max(value, _minimax(state.child(action), maximizing_player_id, alpha, beta, transposition_table))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cutoff
        transposition_table[keys[0]] = [value, alpha, beta, player]
        return value
    else:
        value = float('inf')
        for action in state.legal_actions():
            value = min(value, _minimax(state.child(action), maximizing_player_id, alpha, beta, transposition_table))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff
        transposition_table[keys[0]] = [value, alpha, beta, player]
        return value


def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """

    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    v = _minimax(
        state.clone(),
        maximizing_player_id=maximizing_player_id)
    return v


def main(_):
    global num_rows, num_cols
    # Time how long it takes for game to finish
    start_time = time.time()

    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    num_rows = 3
    num_cols = 3
    game_string = "dots_and_boxes(num_rows=" + str(num_rows) + ",num_cols=" + str(num_cols) + ")"
    print("Creating game: {}".format(game_string))

    game = pyspiel.load_game(game_string)

    value = minimax_search(game)

    if value == 0:
        print("It's a draw")
    else:
        winning_player = 1 if value > 0 else 2
        print(f"Player {winning_player} wins.")

    # Print the time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} milliseconds".format(elapsed_time*1000))
    print("Number of calls to _minimax:", minimax_counter)
    print("Number of times trans table accessed", trans_counter)


if __name__ == "__main__":
    app.run(main)
