# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agents trained on Breakthrough by independent Q-learning."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from open_spiel.python import rl_environment
# from open_spiel.python.algorithms import dqn
import dqn_cnn
from open_spiel.python.algorithms import random_agent
import pyspiel
import os

import time

FLAGS = flags.FLAGS

# Training parameters
package_directory = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(package_directory, 'models/v2cnn')
flags.DEFINE_string("checkpoint_dir", model_file,
                    "Directory to save/load the agent models.")

flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 128, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e6),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    start = time.time()
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]

                # In evaluation, We don't need to add current win_margin info into the time_step
                replaceInfoState(time_step)  # get dbn information out of the observation
                if env.is_turn_based:
                    agent_output = cur_agents[player_id].step(
                        time_step, is_evaluation=True)
                    action_list = [agent_output.action]
                else:
                    agents_output = [
                        agent.step(time_step, is_evaluation=True) for agent in cur_agents
                    ]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)

            sum_episode_rewards[player_pos] += (1 if time_step.rewards[player_pos] > 0 else 0)
    end = time.time()
    logging.info(" Mean episode time consuming %s", (end - start) / num_episodes / num_players)
    return sum_episode_rewards / num_episodes


def main(_):
    num_players = 2
    num_rows, num_cols = 4, 4  # Number of squares
    game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                   "utility_margin=true)")
    game = pyspiel.load_game(game_string)
    env = rl_environment.Environment(game, include_full_state=True)
    """Try to use dbn string instead of observation tensor as the input. Since only the current game layout matters"""
    num_actions = env.action_spec()["num_actions"]
    best_result = [0.8, 0.8]

    # random agents for evaluation
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    agents = [
        dqn_cnn.DQN(
            player_id=idx,
            state_representation_size=(num_rows*2+1,num_cols*2+1,1),
            num_actions=num_actions,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            epsilon_decay_duration=int(1e6),
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]

    for ep in range(FLAGS.num_train_episodes):
        if (ep + 1) % FLAGS.eval_every == 0:
            r_mean = eval_against_random_bots(env, agents, random_agents, 500)
            logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
            # save best result
            for idx in range(num_players):
                if best_result[idx] <= r_mean[idx]:
                    best_result[idx] = r_mean[idx]
                    agents[idx].save(FLAGS.checkpoint_dir)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            # get dbn information out of the observation
            addCurrentWinMargin(time_step)
            replaceInfoState(time_step)
            if env.is_turn_based:
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
            else:
                agents_output = [agent.step(time_step) for agent in agents]
                action_list = [agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            # in the last time step, the win margin is the reward
            time_step.observations["current_margin"] = time_step.rewards
            # print(f"win margin from the last step: {time_step.observations['current_margin']}")
            # get dbn information out of the observation
            replaceInfoState(time_step)
            agent.step(time_step)

        # print(f"Episode {ep} done!")


def replaceInfoState(time_step):
    # get dbn information out of the observation
    game_copy, state_copy = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
    num_rows, num_cols = 4, 4
    dbn = state_copy.dbn_string()
    dbn = np.array(list(dbn), dtype='float')
    h_edges, v_edges = np.split(dbn, [(num_rows + 1) * num_cols])
    h_edges, v_edges = h_edges.reshape((num_rows+1, num_cols)), v_edges.reshape((num_rows, num_cols+1))
    state_info = np.zeros((9,9), dtype='int')
    for i in range(5):
        for j in range(4):
            state_info[2*i, 2*j+1] = h_edges[i,j]
            state_info[2*j+1, 2*i] = v_edges[j,i]

    state_info = state_info[:, :, np.newaxis]
    time_step.observations["info_state"] = [state_info for i in range(2)]


def addCurrentWinMargin(time_step):
    """get the win margin out of the current observation_state"""
    margin4p1 = 0.
    player_id = time_step.observations["current_player"]
    obs_tensor = time_step.observations["info_state"][player_id]

    for row in range(4):
        for col in range(4):
            owner = get_observation_state(obs_tensor, row, col, 'c', as_str=False)
            if owner == 1:
                margin4p1 += 1
            elif owner == 2:
                margin4p1 -= 1

    time_step.observations["current_margin"] = [margin4p1, -margin4p1]
    # game_copy, state_copy = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
    # print(f"Current State[win_margin for is {time_step.observations['current_margin']}]:\n{state_copy}")


num_cols = 4
num_rows = 4
num_cells = (num_rows + 1) * (num_cols + 1)
num_parts = 3  # (horizontal, vertical, cell)
num_states = 3  # (empty, player1, player2)


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
    state = state2num(state)
    part = part2num(part)
    idx = part \
          + (row * (num_cols + 1) + col) * num_parts \
          + state * (num_parts * num_cells)

    return obs_tensor[idx]


def get_observation_state(obs_tensor, row, col, part, as_str=True):
    """
        A helper function intended for getting information of which cells are possessed by which player
    """
    is_state = None
    for state in range(3):
        if get_observation(obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    if as_str:
        is_state = num2state(is_state)
    return is_state


if __name__ == "__main__":
    app.run(main)
