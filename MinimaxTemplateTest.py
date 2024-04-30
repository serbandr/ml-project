import pyspiel
from absl import app
import time

def _minimax(state, maximizing_player_id, alpha=float('-inf'), beta=float('inf'), transposition_table=None):
    """
    Implements a min-max algorithm with transposition tables

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

    # Check if state is terminal or if the player ID is valid
    if state.is_terminal() or not 0 <= state.current_player() < state.num_players() or not 0 <= maximizing_player_id < state.num_players():
        return state.player_return(maximizing_player_id)

    if transposition_table is None:
        transposition_table = {}

    # Observation tensor is the key (like '1111...000')
    key = ''.join('1' if val == 1.0 else '0' for val in state.observation_tensor())

    if key in transposition_table:
        return transposition_table[key]

    # Use Alpha-Beta pruning to speed up the process
    player = state.current_player()
    if player == maximizing_player_id:
        value = float('-inf')
        for action in state.legal_actions():
            value = max(value, _minimax(state.child(action), maximizing_player_id, alpha, beta, transposition_table))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        transposition_table[key] = value
        return value
    else:
        value = float('inf')
        for action in state.legal_actions():
            value = min(value, _minimax(state.child(action), maximizing_player_id, alpha, beta, transposition_table))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff
        transposition_table[key] = value
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
    game_string = "dots_and_boxes(num_rows=2,num_cols=2)"
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

if __name__ == "__main__":
    app.run(main)
