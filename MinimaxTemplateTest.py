import pyspiel
from absl import app
import time
import re
import numpy as np
from collections import Counter

minimax_counter = 0
trans_counter = 0

def _symmetric_key(state):
    """Generate symmetric keys for the given state."""

    # Get number of cols and rows
    num_cols = 0
    num_rows = 0
    game_string = str(state.get_game())
    match = re.search(r'num_cols=(\d+),num_rows=(\d+)', game_string)
    if match:
        num_cols = int(match.group(1))
        num_rows = int(match.group(2))

    # Use the dbns!

    dbns = state.dbn_string()

    num_horiz_lines = num_cols * (1 + num_rows)
    formatted_dbn_string = [dbns[0:num_horiz_lines], dbns[num_horiz_lines:]]

    horiz_string = formatted_dbn_string[0]
    split_array = [''.join(horiz_string[i:i + num_cols]) for i in range(0, len(horiz_string), num_cols)]
    formatted_dbn_string[0] = split_array

    vert_string = formatted_dbn_string[1]
    split_array = [''.join(vert_string[i:i + num_rows+1]) for i in range(0, len(vert_string), num_rows+1)]
    split_array_formatted = []
    for i in range(len(split_array[0])):
        split_array_formatted.append(''.join([elem[i] for elem in split_array]))
    formatted_dbn_string[1] = split_array_formatted

    # print("DBN String")
    # print(formatted_dbn_string)

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
    # print("Symmetries")
    # print([horiz_lines, vert_lines])

    # TODO : If it's a square game, add diagonal symmetry

    # # Assuming state.observation_tensor() returns a flat list
    # observation_tensor = np.array(state.observation_tensor()).reshape(12, 12)
    # # Vertical symmetry
    #
    # keys = [''.join('1' if val == 1.0 else '0' for val in observation_tensor.flat),
    #         ''.join('1' if val == 1.0 else '0' for val in np.flip(observation_tensor, axis=0).flat),
    #         ''.join('1' if val == 1.0 else '0' for val in np.flip(observation_tensor, axis=1).flat),
    #         ''.join('1' if val == 1.0 else '0' for val in np.flip(np.flip(observation_tensor, axis=0), axis=1).flat)]
    # keys = [''.join('1' if val == 1.0 else '0' for val in state.observation_tensor())]

    return keys


def _minimax(state, maximizing_player_id, alpha=float('-inf'), beta=float('inf'), transposition_table=None):
    """
    Implements a min-max algorithm with transposition tables and move ordering.

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

    # print(state.dbn_string())
    # print(state)

    global minimax_counter, trans_counter
    minimax_counter += 1

    # Check if state is terminal or if the player ID is valid
    if state.is_terminal() or not 0 <= state.current_player() < state.num_players() or not 0 <= maximizing_player_id < state.num_players():
        # TODO : Win margin instead of just plain win
        return state.player_return(maximizing_player_id)

    if transposition_table is None:
        transposition_table = {}

    player = state.current_player()

    # Evaluate moves and order them based on evaluation scores
    # moves = [(action, evaluate_move(state, action, player)) for action in state.legal_actions()]
    # moves.sort(key=lambda x: x[1], reverse=True)  # Order moves in descending order of evaluation score

    # Observation tensor is the key (like '1111...000')
    keys = _symmetric_key(state)

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
            if alpha >= beta:
                break  # Beta cutoff
        for key in keys:
            transposition_table[key] = [value, alpha, beta, player]
        return value
    else:
        value = float('inf')
        for action in state.legal_actions():
            value = min(value, _minimax(state.child(action), maximizing_player_id, alpha, beta, transposition_table))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff
        for key in keys:
            transposition_table[key] = [value, alpha, beta, player]
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
    # Time how long it takes for game to finish
    start_time = time.time()

    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = "dots_and_boxes(num_rows=4,num_cols=4)"
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
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    print("Number of calls to _minimax:", minimax_counter)
    print("Number of times trans table accessed", trans_counter)


if __name__ == "__main__":
    app.run(main)
