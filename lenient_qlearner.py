from math import exp

from open_spiel.python.algorithms import tabular_qlearner

"""Tabular Q-learning agent.(Review)"""

import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools

"""
The Q value table/dict:
  1. It's a nest dict:
    outer: state
    inner: action
  2. default_dict:
  allow use to set default values of a new key efficiently
    for outer: default value is a dict(action -> q-value(float))
    for inner: default value is a float(float()=0.)
"""


def valuedict():
    # The default factory is called without arguments to produce a new value when
    # a key is not present, in __getitem__ only. This value is added to the dict,
    # so modifying it will modify the dict.
    return collections.defaultdict(float)

def rewarddict():
    # default dictionary for recording K rewards
    return collections.defaultdict(list)


class LFAQlearner(rl_agent.AbstractAgent):
    """
    LFAQ learner based on the paper: https://www.researchgate.net/publication/221454203_Frequency_adjusted_multiagent_Q-learning
    """
    def __init__(self,
                 player_id,
                 num_actions,
                 step_size_a=0.05,  # learning rate: α for FAQ
                 step_size_b=0.001,  # β for FAQ
                 temperature_schedule=rl_tools.ConstantSchedule(0.2),
                 K=3,  # Leniency
                 discount_factor=1.0,
                 centralized=False):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size_a = step_size_a
        self._step_size_b = step_size_b
        self._K = K
        self._temperature_schedule = temperature_schedule
        self._temperature = temperature_schedule.value
        self._discount_factor = discount_factor
        self._centralized = centralized
        self._q_values = collections.defaultdict(valuedict)
        self._rewards = collections.defaultdict(rewarddict)
        self._prev_info_state = None
        self._last_loss_value = None

    def _get_action_probs(self, info_state, legal_actions, temperature):
        """Returns a selected action and the probabilities of legal actions.

        To be overwritten by subclasses that implement other action selection
        methods.

        Args:
          info_state: hashable representation of the information state.
          legal_actions: list of actions at `info_state`.
          temperature: exploration parameter for boltzmann exploration
        """
        return self._boltzmann(info_state, legal_actions, temperature)

    def _boltzmann(self, info_state, legal_actions, temperature):
        """
        Generate policy based on boltzmann exploration
        :param info_state: current state
        :param legal_actions: legal actions for current state
        :param temperature: exploration parameter
        :return: chosen action, probability distribution(policy)
        """

        # using info state as the current state / part of the key to access q value
        processed_Q = np.array([exp(self._q_values[info_state][a] / temperature) for a in legal_actions])

        sum_Q = np.sum(processed_Q)

        probs = processed_Q / sum_Q

        # randomly choose one
        action = np.random.choice(range(self._num_actions), p=probs)
        # return the chosen action and current probability distribution
        return action, probs

    def step(self, time_step, is_evaluation=False):

        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        if self._centralized:
            info_state = str(time_step.observations["info_state"])
        else:
            info_state = str(time_step.observations["info_state"][self._player_id])

        # legal actions is told by the environment
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            temperature = 0.0 if is_evaluation else self._temperature
            action, probs = self._get_action_probs(info_state, legal_actions, temperature)


        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            """
            Reward of the last action, which takes the agent from the previous state to the current one : r
            """
            target = time_step.rewards[self._player_id]

            """
            Note down current reward until hit the leniency limit
            """
            rewards = self._rewards[self._prev_info_state][self._prev_action]
            rewards.append(target)

            # update Q value after hitting the leniency limit
            if len(rewards) == self._K:
                # choose the maximum reward
                target = max(rewards)
                # clear the note
                del self._rewards[self._prev_info_state][self._prev_action]

                """
                target(previous_state, previous_action) = r + discount_factor * max(Q of current state)
                """
                if not time_step.last():  # Q values are zero for terminal states.
                    target += self._discount_factor * max(
                        [self._q_values[info_state][a] for a in legal_actions])

                """
                Q_{t-1}(previous_state, previous_action)
                """
                prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
                # get difference: update part without multiplying by the learning rate
                self._last_loss_value = target - prev_q_value

                """
                FAQ : https://www.researchgate.net/publication/221454203_Frequency_adjusted_multiagent_Q-learning
                update q-value(step_size is learning rate here):
                Q_{t} = Q_{t-1} + min(B/last_x,1) * A * last_loss
                """
                self._q_values[self._prev_info_state][self._prev_action] += (
                    min(self._step_size_b / self._prev_action_prob , 1) * self._step_size_a * self._last_loss_value
                )

            # Decay epsilon, if necessary.
            self._temperature = self._temperature_schedule.step()

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
            self._prev_action_prob = probs[action]

        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value

"""
Below are some interpretations of the source code for the Qlearner, provided by OpenSpiel, written by me for learning purposes
"""
#
# class QLearner(rl_agent.AbstractAgent):
#   """Tabular Q-Learning agent.
#
#   See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
#   """
#
#   def __init__(self,
#                player_id,
#                num_actions,
#                step_size=0.1,
#                epsilon_schedule=rl_tools.ConstantSchedule(0.2),
#                discount_factor=1.0,
#                centralized=False):
#     """Initialize the Q-Learning agent."""
#     self._player_id = player_id
#     self._num_actions = num_actions
#     self._step_size = step_size
#     """
#       the parameter includes a epsilon_schedule rather than simply an epsilon value since we take decay of epsilon into account
#     """
#     self._epsilon_schedule = epsilon_schedule
#     self._epsilon = epsilon_schedule.value
#     self._discount_factor = discount_factor
#     self._centralized = centralized
#     self._q_values = collections.defaultdict(valuedict)
#     self._prev_info_state = None
#     self._last_loss_value = None
#
#
#
#   def _epsilon_greedy(self, info_state, legal_actions, epsilon):
#     """Returns a valid epsilon-greedy action and valid action probs.
#
#     If the agent has not been to `info_state`, a valid random action is chosen.
#
#     Args:
#       info_state: hashable representation of the information state.
#       legal_actions: list of actions at `info_state`.
#       epsilon: float, prob of taking an exploratory action.
#
#     Returns:
#       A valid epsilon-greedy action and valid action probabilities.
#     """
#     # set the probability of each action to be zero
#     # even though, we only consider legal actions, but we still need to return the probability of those illegal actions(which is supposed to be zero)
#     probs = np.zeros(self._num_actions)
#     # using info state as the current state / part of the key to access q value
#     greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
#
#     """as a result of using default dict, we don't have to worry about if the given key exists. If not, it will create its
#     default value and return it"""
#     # extract the greedy actions(policy actions), since there might be multiple optimal actions(with the same q-value)
#     greedy_actions = [
#         a for a in legal_actions if self._q_values[info_state][a] == greedy_q
#     ]
#
#     # exploration(if epsilon == 0, then the agent only chooses the optimal action)
#     probs[legal_actions] = epsilon / len(legal_actions)
#     # exploitation(For multiple optimal actions, give them the equal probability)
#     probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
#     # randomly choose one
#     action = np.random.choice(range(self._num_actions), p=probs)
#     # return the chosen action and current probability distribution
#     return action, probs
#
#   def _get_action_probs(self, info_state, legal_actions, epsilon):
#     """Returns a selected action and the probabilities of legal actions.
#
#     To be overwritten by subclasses that implement other action selection
#     methods.
#
#     Args:
#       info_state: hashable representation of the information state.
#       legal_actions: list of actions at `info_state`.
#       epsilon: float: current value of the epsilon schedule or 0 in case
#         evaluation. QLearner uses it as the exploration parameter in
#         epsilon-greedy, but subclasses are free to interpret in different ways
#         (e.g. as temperature in softmax).
#     """
#     return self._epsilon_greedy(info_state, legal_actions, epsilon)
#
#   def step(self, time_step, is_evaluation=False):
#     """Returns the action to be taken and updates the Q-values if needed.
#
#     Args:
#       time_step: an instance of rl_environment.TimeStep.
#       is_evaluation: bool, whether this is a training or evaluation call.
#
#     Returns:
#       A `rl_agent.StepOutput` containing the action probs and chosen action.
#     """
#     if self._centralized:
#       info_state = str(time_step.observations["info_state"])
#     else:
#       info_state = str(time_step.observations["info_state"][self._player_id])
#
#     # legal actions is told by the environment
#     legal_actions = time_step.observations["legal_actions"][self._player_id]
#
#     # Prevent undefined errors if this agent never plays until terminal step
#     action, probs = None, None
#
#     # Act step: don't act at terminal states.
#     if not time_step.last():
#       epsilon = 0.0 if is_evaluation else self._epsilon
#       action, probs = self._get_action_probs(info_state, legal_actions, epsilon)
#
#
#     # Learn step: don't learn during evaluation or at first agent steps.
#     # the self._prev_info_state is none at the first call of step
#     if self._prev_info_state and not is_evaluation:
#       """
#       Reward of the last action, which takes the agent frm previous state to the current one : r
#       """
#       target = time_step.rewards[self._player_id]
#
#       """
#       target(previous_state, previous_action) = r + discount_factor * max(Q of current state)
#       """
#       if not time_step.last():  # Q values are zero for terminal states.
#         target += self._discount_factor * max(
#             [self._q_values[info_state][a] for a in legal_actions])
#
#       """
#       Q_{t-1}(previous_state, previous_action)
#       """
#       prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
#       # get difference: update part without multiplying by the learning rate
#       self._last_loss_value = target - prev_q_value
#       # update q-value(step_size is learning rate here)
#       self._q_values[self._prev_info_state][self._prev_action] += (
#           self._step_size * self._last_loss_value)
#
#       # Decay epsilon, if necessary.
#       self._epsilon = self._epsilon_schedule.step()
#
#       if time_step.last():  # prepare for the next episode.
#         self._prev_info_state = None
#         return
#
#     # Don't mess up with the state during evaluation.
#     if not is_evaluation:
#       self._prev_info_state = info_state
#       self._prev_action = action
#     return rl_agent.StepOutput(action=action, probs=probs)
#
#   @property
#   def loss(self):
#     return self._last_loss_value
#
