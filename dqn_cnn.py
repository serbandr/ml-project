""" A copy of dpn """

"""DQN agent implemented in TensorFlow."""

import collections
import os
from absl import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError


from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask current_margin next_margin")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


def cnn(input_size, output_size):
    """
     This CNN is tailored for the dots&boxes game
    :param input_size:
    :return:
    """
    input = Input(shape=input_size)
    """
    Convolut over the whole game board with size 3x3 and stride 2. This allows the kernel to be positioned at each box.
    Ideally it can give some information about whether one block can be captured immediately by gripping one specific edge of the four(Thus using four filters).
    """
    x = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="valid")(input)
    # The following layers are supposed to exploit the chains
    x = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="valid")(x)
    # Flatten
    x = Flatten()(x)
    output = Dense(output_size)(x)

    return tf.keras.Model(inputs=input, outputs=output)




class DQN(rl_agent.AbstractAgent):
    """DQN Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 state_representation_size,
                 num_actions,
                 replay_buffer_capacity=10000,
                 batch_size=128,
                 replay_buffer_class=ReplayBuffer,
                 learning_rate=0.01,
                 update_target_network_every=1000,
                 learn_every=10,
                 discount_factor=1.0,
                 min_buffer_size_to_learn=1000,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_duration=int(1e6),
                 optimizer_str="sgd"):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()

        self.player_id = player_id
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._update_target_network_every = update_target_network_every
        self._learn_every = learn_every
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._discount_factor = discount_factor

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration

        if not isinstance(replay_buffer_capacity, int):
            raise ValueError("Replay buffer capacity not an integer.")
        # create a replayer buffer with the specified size
        self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None

        # Step counter to keep track of learning, eps decay and target network.
        self._step_counter = 0

        # Keep track of the last training loss achieved in an update step.
        self._last_loss_value = None

        if optimizer_str == "adam":
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_str == "sgd":
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")



        """Deep Neural Network for actual learning"""
        self._q_network = cnn(state_representation_size, num_actions)

        """Target Deep Neural Network for generating target q-value"""
        self._target_q_network = cnn(state_representation_size, num_actions)



    def get_step_counter(self):
        return self._step_counter

    def step(self, time_step, is_evaluation=False, add_transition_record=True):
        """Returns the action to be taken and updates the Q-network if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.
          add_transition_record: Whether to add to the replay buffer on this step.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """

        # Act step: don't act at terminal info states or if it's not our turn.
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            epsilon = self._get_epsilon(is_evaluation)
            action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon)
        else:
            action = None
            probs = []

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._step_counter += 1

            if self._step_counter % self._learn_every == 0:
                self._last_loss_value = self.learn()

            """Update the weights of target_network"""
            if self._step_counter % self._update_target_network_every == 0:
                self._target_q_network.set_weights(self._q_network.get_weights())

            if self._prev_timestep and add_transition_record:
                # We may omit record adding here if it's done elsewhere.
                self.add_transition(self._prev_timestep, self._prev_action, time_step)

            if time_step.last():  # prepare for the next episode.
                self._prev_timestep = None
                self._prev_action = None
                return
            else:
                self._prev_timestep = time_step
                self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_time_step, prev_action, time_step):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """
        assert prev_time_step is not None
        legal_actions = (time_step.observations["legal_actions"][self.player_id])
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            info_state=(
                prev_time_step.observations["info_state"][self.player_id][:]),
            action=prev_action,
            reward=time_step.rewards[self.player_id],
            next_info_state=time_step.observations["info_state"][self.player_id][:],
            is_final_step=float(time_step.last()),
            legal_actions_mask=legal_actions_mask,
            current_margin=prev_time_step.observations["current_margin"][self.player_id],
            next_margin=time_step.observations["current_margin"][self.player_id]
        )
        self._replay_buffer.add(transition)

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        """Returns a valid epsilon-greedy action and valid action probs.

        Action probabilities are given by a softmax over legal q-values.

        Args:
          info_state: hashable representation of the information state.
          legal_actions: list of legal actions at `info_state`.
          epsilon: float, probability of taking an exploratory action.

        Returns:
          A valid epsilon-greedy action and valid action probabilities.
        """
        probs = np.zeros(self._num_actions)
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            info_state = info_state[np.newaxis, :]
            """Give concrete input and get predicted Q-values(only do prediction)"""
            # print(f"info_state: {info_state}")
            # print(info_state.shape)
            q_values = self._q_network(info_state)[0].numpy()
            # print(f"legal actions: {legal_actions}")
            legal_q_values = q_values[legal_actions]
            # print(f"legal q values: {legal_q_values}")
            action = legal_actions[np.argmax(legal_q_values)]
            probs[action] = 1.0
        return action, probs

    def _get_epsilon(self, is_evaluation, power=1.0):
        """Returns the evaluation or decayed epsilon value."""
        if is_evaluation:
            return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
                self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
                (1 - decay_steps / self._epsilon_decay_duration) ** power)
        return decayed_epsilon

    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """

        if (len(self._replay_buffer) < self._batch_size or
                len(self._replay_buffer) < self._min_buffer_size_to_learn):
            return None

        transitions = self._replay_buffer.sample(self._batch_size)
        info_states = np.array([t.info_state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        current_margins = np.array([t.current_margin for t in transitions])
        next_margins = np.array([t.next_margin for t in transitions])
        next_info_states = np.array([t.next_info_state for t in transitions])
        # 1 for all that are final states
        are_final_steps = np.array([t.is_final_step for t in transitions])
        legal_actions_mask = np.array([t.legal_actions_mask for t in transitions])


        """The illegal logits gives a extremely large negative number so that our agent will never choose the illegal actions"""
        # all illegal actions will be 1
        illegal_actions = 1 - legal_actions_mask
        illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

        # predict q-values based on the target network
        target_q_values = self._target_q_network(next_info_states)

        """For each sample, get the maximum q-value of the next state"""
        max_next_q = tf.reduce_max(
            tf.math.add(target_q_values, illegal_logits),
            axis=-1)

        """Calculate the target q-value(These are only for one certain action of one state)"""
        target = (
                rewards +
                (1 - are_final_steps) * self._discount_factor * (max_next_q + next_margins) -
                current_margins
        )

        # target = (
        #         rewards +
        #         (1 - are_final_steps) * self._discount_factor * max_next_q
        # )

        """Optimizing model with the manual loss function"""
        with tf.GradientTape() as tape:
            # predict q_values
            q_values = self._q_network(info_states)

            # extracting corresponding predicted values
            """"By stacking, we assign action to every sample"""
            action_indices = tf.stack(
                [tf.range(tf.shape(q_values)[0]), actions], axis=-1)
            """We get the predicted value of these state-action pairs"""
            predictions = tf.gather_nd(q_values, action_indices)

            # calculate mean loss among the dataset
            loss_func = MeanSquaredError()
            loss = loss_func(target, predictions)

        # optimize the network
        self._optimizer.minimize(loss, self._q_network.trainable_variables, tape=tape)


    def _full_checkpoint_name(self, checkpoint_dir, name):
        checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
        return os.path.join(checkpoint_dir, checkpoint_filename)


    def save(self, checkpoint_dir):
        """Saves the q network and the target q-network.

        Note that this does not save the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory where checkpoints will be saved.
        """
        self._q_network.save_weights(self._full_checkpoint_name(checkpoint_dir, "q_network"))
        self._target_q_network.save_weights(self._full_checkpoint_name(checkpoint_dir, "target_q_network"))
        logging.info("Saved to path: %s", checkpoint_dir)

    def restore(self, checkpoint_dir):
        """Restores the q network and the target q-network.

        Note that this does not restore the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory from which checkpoints will be restored.
        """
        self._q_network.load_weights(self._full_checkpoint_name(checkpoint_dir, "q_network"))
        self._target_q_network.load_weights(self._full_checkpoint_name(checkpoint_dir, "target_q_network"))
        logging.info("Restoring checkpoint: %s", checkpoint_dir)

    """
        We are able to access the raw q-value after using step
    """

    # @property
    # def q_values(self):
    #     return self._q_values
    #
    # @property
    # def replay_buffer(self):
    #     return self._replay_buffer
    #
    # @property
    # def info_state_ph(self):
    #     return self._info_state_ph
    #
    # @property
    # def loss(self):
    #     return self._last_loss_value
    #
    # @property
    # def prev_timestep(self):
    #     return self._prev_timestep
    #
    # @property
    # def prev_action(self):
    #     return self._prev_action
    #
    # @property
    # def step_counter(self):
    #     return self._step_counter


