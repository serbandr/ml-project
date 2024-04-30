from absl import app
from absl import flags
import numpy as np
import pyspiel

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random

FLAGS = flags.FLAGS

# Supported types of players: "random", "human"
flags.DEFINE_string("player0", "random", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")

def LoadAgent(agent_type, player_id, rng):
    """Return a bot based on the agent type."""
    if agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "human":
        return human.HumanBot()
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))

def main(_):
    rng = np.random.RandomState()

    num_rows, num_cols = 2, 2  # Number of squares
    game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                   "utility_margin=true)")
    game = pyspiel.load_game(game_string)

    # Set agents

    agents = [
        LoadAgent(FLAGS.player0, 0, rng),
        LoadAgent(FLAGS.player1, 1, rng),
    ]

    # Play game
    state = game.new_initial_state()

    # Print the initial state
    print("INITIAL STATE")
    print(str(state))

    while not state.is_terminal():
        current_player = state.current_player()

        # Decision node: sample action for the single current player
        legal_actions = state.legal_actions()
        for action in legal_actions:
            print(
                "Legal action: {} ({})".format(
                    state.action_to_string(current_player, action), action
                )
            )

        action = agents[current_player].step(state)
        action_string = state.action_to_string(current_player, action)
        print("Player ", current_player+1, ", chose action: ", action_string)
        state.apply_action(action)

        print("")
        print("NEXT STATE:")
        print(str(state))
        if not state.is_terminal():
            print(str(state.observation_tensor()))

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid+1, returns[pid]))

app.run(main)