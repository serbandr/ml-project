#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent_v2.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots

from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import TimeStep
import dqn_cnn
import os
import random


logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self._dqn_agent = dqn_cnn.DQN(
            player_id=self.player_id,
            state_representation_size=(9,9,1),
            num_actions=40,
        )

        # weights directory
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_directory = os.path.join(package_directory, 'models/v2cnn')
        # restore weights
        self._dqn_agent.restore(checkpoint_directory)

        game_string = (f"dots_and_boxes(num_rows={4},num_cols={4},"
                       "utility_margin=true)")
        # 4x4 game used for generating window state
        self._game4x4 = pyspiel.load_game(game_string)

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """

        if state.is_terminal():
            return None

        # get the board size
        game = state.get_game()
        game_paras = game.get_parameters()
        num_rows, num_cols = game_paras["num_rows"], game_paras["num_cols"]

        # dbn_string representation of the current game state
        dbn = state.dbn_string()
        dbn = np.array(list(dbn), dtype='int')
        # split the dbn string to horizontal & vertical edges
        h_edges, v_edges = np.split(dbn, [(num_rows + 1) * num_cols])

        # padding the h_edges and v_edges
        nr = max(num_rows, 4)
        nc = max(num_cols, 4)
        padded_h_edges = np.ones((nr + 1, nc), dtype='int')
        padded_h_edges[:num_rows + 1, :num_cols] = h_edges.reshape(num_rows + 1, num_cols)

        padded_v_edges = np.ones((nr, nc + 1), dtype='int')
        padded_v_edges[:num_rows, :num_cols + 1] = v_edges.reshape(num_rows, num_cols + 1)

        h_edges, v_edges = padded_h_edges, padded_v_edges

        """Using 4x4 window to convolut over the whole game board(first version with stride = 1)"""
        best_actions = {}
        # print(f"----Original_state----:\n{state}")
        for r_offset in range(nr - 4 + 1):
            for c_offset in range(nc - 4 + 1):
                h_window = h_edges[r_offset: r_offset + 5, c_offset: c_offset + 4]
                v_window = v_edges[r_offset: r_offset + 4, c_offset: c_offset + 5]

                # constructing info state
                state_info = np.zeros((9, 9), dtype='int')
                for i in range(5):
                    for j in range(4):
                        state_info[2 * i, 2 * j + 1] = h_window[i, j]
                        state_info[2 * j + 1, 2 * i] = v_window[j, i]
                state_info = state_info[:, :, np.newaxis]

                # construct the window state to get legal actions
                h_window = h_window.reshape([-1])
                v_window = v_window.reshape([-1])
                window_dbn = np.concatenate([h_window, v_window])
                window_dbn_str = ''.join(window_dbn.astype('str'))
                window_state = self._game4x4.new_initial_state(window_dbn_str)

                # skip current block, if there's no possible moves.
                if len(window_state.legal_actions()) == 0:
                    continue

                # Constructing TimeStep which is used by the agent
                observations = {
                    "info_state": [],
                    "legal_actions": [],
                    "current_player": [],
                    "serialized_state": []
                }

                for player_id in range(2):
                    observations["info_state"].append(state_info)   # dbn is used as the info_state
                    observations["legal_actions"].append(window_state.legal_actions())
                observations['current_player']=state.current_player()
                # print(f"window_state({r_offset},{c_offset}):\n{window_state}", end="")
                time_step = TimeStep(
                    observations=observations,
                    # observation is the only thing we need
                    rewards=None,
                    discounts=None,
                    step_type=rl_environment.StepType.MID # since we already skipped the last step, we can always set it mid
                )
                # get the best action along with it's expected q-value
                output = self._dqn_agent.step(time_step, is_evaluation=True)
                action = output.action
                # print(f"action {action} -> ", end="")

                # Mapping the action back to the padded board
                if action < 20:  # horizontal edges
                    action = action + c_offset + r_offset * nc + action // 4 * (nc - 4)
                else:
                    action = (nc * (nr + 1) - 20) + action + c_offset + r_offset * (nc + 1) + (
                                action - 20) // 5 * (nc - 4)
                # print(action)
                # update the dictionary for action-q_value pair
                if action not in best_actions.keys():
                    best_actions[action] = 0
                best_actions[action] += 1
        # print(best_actions)
        # TODO: trying different optimal policy choosing methods
        """
            Current method: majority vote among all the window games
            For example, get the q_value and adding them up? 
        """
        optimal_val = best_actions.get(max(best_actions, key=best_actions.get))
        # randomly pick up one among the best actions.
        optimal_action = random.choice([k for k, v in best_actions.items() if v == optimal_val])

        # Mapping the action on the padded board back to the original board
        if optimal_action < (nr + 1) * nc:  # horizontal edge
            optimal_action = optimal_action // nc * num_cols + optimal_action % nc
        else:
            offset = optimal_action - (nr + 1) * nc
            optimal_action = (num_rows + 1) * num_cols + offset // (nc + 1) * (num_cols + 1) + offset % (nc + 1)

        # print(optimal_action)
        # print(optimal_action in state.legal_actions())
        return optimal_action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=5,num_cols=5)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
