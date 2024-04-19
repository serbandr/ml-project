from open_spiel.python.egt import dynamics
import numpy as np
import math

"""
Since the leniency mechanism only changes the way to compute fitness. We shall change Single/Multi PopulationDynamics 
rather than boltzmannq or replicator.

Parameter Tuning:
No need to introduce learning rate. On the one hand, it only changes magnitude; On the other hand, boltzmannq doesn't introduce it either.
"""

def leniency4Matrix(state, payoff_matrix, K):
    """
    Calculating fitness for matrix game with leniency mechanism
    :param payoff_matrix:
    :param state:
    :param K:
    :return:
    """
    n_actions_x = payoff_matrix.shape[0]
    n_actions_y = payoff_matrix.shape[1]
    fitness = np.zeros(n_actions_x)
    # calculate expected utility for each action
    for i in range(n_actions_x):
        for j in range(n_actions_y):
            payoff = payoff_matrix[i,j]
            less = np.sum(state[payoff_matrix[i] < payoff])
            equal = np.sum(state[payoff_matrix[i] == payoff])
            fitness[i] += (payoff * state[j] * (math.pow(less + equal, K) - math.pow(less, K)) / equal)
    return fitness

class LBQSinglePopulationDynamics(dynamics.SinglePopulationDynamics):
    def __init__(self, payoff_matrix, K=10, T=1.):
        """Initializes the single-population dynamics."""
        super().__init__(payoff_matrix, dynamics.boltzmannq)
        self.K = K
        self.T = T

    def __call__(self, state=None, time=None):
        """Time derivative of the population state.

        Args:
          state: Probability distribution as list or
            `numpy.ndarray(shape=num_strategies)`.
          time: Time is ignored (time-invariant dynamics). Including the argument in
            the function signature supports numerical integration via e.g.
            `scipy.integrate.odeint` which requires that the callback function has
            at least two arguments (state and time).

        Returns:
          Time derivative of the population state as
          `numpy.ndarray(shape=num_strategies)`.
        """
        state = np.array(state)
        assert state.ndim == 1
        assert state.shape[0] == self.payoff_matrix.shape[0]
        fitness = leniency4Matrix(state, self.payoff_matrix, self.K)
        return self.dynamics(state, fitness, temperature=self.T)


class LBQTwoPopulationDynamics(dynamics.MultiPopulationDynamics):
    def __init__(self, payoff_tensor, K=10, T=1.):
        """Initializes the multi-population dynamics."""
        super().__init__(payoff_tensor, dynamics.boltzmannq)
        assert payoff_tensor.shape[0] == 2
        self.K = K
        self.T = T

    def __call__(self, state, time=None):
        """Time derivative of the population states.

        Args:
          state: Combined population state for all populations as a list or flat
            `numpy.ndarray` (ndim=1). Probability distributions are concatenated in
            order of the players.
          time: Time is ignored (time-invariant dynamics). Including the argument in
            the function signature supports numerical integration via e.g.
            `scipy.integrate.odeint` which requires that the callback function has
            at least two arguments (state and time).

        Returns:
          Time derivative of the combined population state as `numpy.ndarray`.
        """
        state = np.array(state)
        ks = self.payoff_tensor.shape[1:]  # number of strategies for each player

        assert state.shape[0] == sum(ks)

        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * 2
        for i in range(2):
            # move i-th population to front(it is equal to transpose the matrix for the second player)
            # It's the payoff matrix for the current player
            fitness = np.moveaxis(self.payoff_tensor[i], i, 0)
            fitness = leniency4Matrix(states[1-i], fitness, self.K)
            dstates[i] = self.dynamics[i](states[i], fitness, self.T)

        return np.concatenate(dstates)
