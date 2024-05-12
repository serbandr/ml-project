from absl import app
from absl import flags
import numpy as np
import pyspiel
import time
import tensorflow as tf

from open_spiel.python.algorithms.mcts import Evaluator

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import mcts_agent

FLAGS = flags.FLAGS

# Supported types of players: "random", "human"
flags.DEFINE_string("player0", "mcts", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")


# Make my own evaluator for MCTS

class NeuralNetworkEvaluator:
    def __init__(self, game, model):
        self.game = game
        self.model = model

    def evaluate(self, state):
        features = self.extract_features(state)
        value, policy = self.model.predict(features)
        return value[0], policy[0]

    def prior(self, state):
        legal_actions = state.legal_actions()
        features = self.extract_features(state)
        _, policy = self.model.predict(features)
        policy = policy[0][legal_actions]  # Only consider legal actions
        return [(action, prob) for action, prob in zip(legal_actions, policy)]

    def extract_features(self, state):
        # Extract features from the state (e.g., board representation)
        # Convert state to a format suitable for input to the neural network
        return np.array(features)

class NeuralNetworkModel(tf.keras.Model):
    def __init__(self, input_shape, output_shape_value, output_shape_policy):
        super(NeuralNetworkModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.value_output_layer = tf.keras.layers.Dense(output_shape_value, activation='linear')
        self.policy_output_layer = tf.keras.layers.Dense(output_shape_policy, activation='softmax')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        value = self.value_output_layer(x)
        policy = self.policy_output_layer(x)
        return value, policy

class MyEvaluator(Evaluator):
    """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

    def __init__(self, n_rollouts=1, random_state=None):
        self.n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()

    def evaluate(self, state):
        """Returns evaluation on given state."""
        result = None
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            while not working_state.is_terminal():
                if working_state.is_chance_node():
                    outcomes = working_state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = self._random_state.choice(action_list, p=prob_list)
                else:
                    action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
            returns = np.array(working_state.returns())
            result = returns if result is None else result + returns

        return result / self.n_rollouts

    def prior(self, state):
        """Returns equal probability for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


def LoadAgent(agent_type, player_id, rng, game_string=None):
    """Return a bot based on the agent type."""
    if agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "human":
        return human.HumanBot()
    elif agent_type == "mcts":
        env = rl_environment.Environment(game_string, include_full_state=True)
        num_actions = env.action_spec()["num_actions"]
        mcts_bot = mcts.MCTSBot(env.game, 1.5, 100, MyEvaluator())
        return mcts_bot
        # return mcts_agent.MCTSAgent(player_id=player_id, num_actions=num_actions, mcts_bot=mcts_bot)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def main(_):
    rng = np.random.RandomState()

    num_rows, num_cols = 7, 7  # Number of squares
    game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                   "utility_margin=true)")
    game = pyspiel.load_game(game_string)

    # Set agents

    agents = [
        LoadAgent(FLAGS.player0, 0, rng, game_string),
        LoadAgent(FLAGS.player1, 1, rng),
    ]

    # Play game
    state = game.new_initial_state()

    # Print the initial state
    # print("INITIAL STATE")
    # print(str(state))

    win_p1 = 0
    win_p2 = 0

    num_games = 1

    for i in range(0, num_games):

        while not state.is_terminal():
            current_player = state.current_player()

            # Decision node: sample action for the single current player
            legal_actions = state.legal_actions()
            # for action in legal_actions:
            # print(
            #    "Legal action: {} ({})".format(
            #        state.action_to_string(current_player, action), action
            #    )
            # )

            start_time = time.time()
            action = agents[current_player].step(state)
            end_time = time.time()
            print(end_time - start_time)
            action_string = state.action_to_string(current_player, action)
            # print("Player ", current_player + 1, ", chose action: ", action_string)
            state.apply_action(action)

            # print("")
            # print("NEXT STATE:")
            # print(str(state))
            # if not state.is_terminal():
            #    print(str(state.observation_tensor()))

        # Game is now done. Print utilities for each player
        returns = state.returns()
        # for pid in range(game.num_players()):
        #    print("Utility for player {} is {}".format(pid + 1, returns[pid]))

        if returns[0] > 0:
            win_p1 += 1
        else:
            win_p2 += 1

    print("Player 1")
    print(win_p1)
    print("Player 2")
    print(win_p2)


app.run(main)
