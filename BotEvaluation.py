import pyspiel
import evaluate_bots
from open_spiel.python.bots import uniform_random
from dotsandboxes_agent_v2 import get_agent_for_tournament
# import dotsandboxes_agent
import numpy as np
import time
import first_openedge


def eval_against_random_bots(trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_rows, num_cols = 4, 4  # Number of squares
    game_string = (f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols},"
                   "utility_margin=true)")
    game = pyspiel.load_game(game_string)

    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    start = time.time()
    for player_pos in range(num_players):
        print(f"Evaluating agent as {player_pos}:")
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for episode in range(num_episodes):
            if episode % 20 == 0:
                print(f"Episode {episode}")
            episode_rewards = evaluate_bots.evaluate_bots(game.new_initial_state(), cur_agents, np.random)
            print(episode_rewards)
            sum_episode_rewards[player_pos] += (1 if episode_rewards[player_pos] > 0 else 0)
    end = time.time()
    print(" Mean episode time: %s",  (end-start)/num_episodes/num_players)
    return sum_episode_rewards / num_episodes

if __name__ == "__main__":
    num_players = 2
    random_bots = [
        uniform_random.UniformRandomBot(player_id=idx, rng=np.random)
        # first_openedge.FirstOpenEdge(player_id=idx)
        # dotsandboxes_agent.get_agent_for_tournament(idx)
        for idx in range(num_players)
    ]
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    print(eval_against_random_bots(bots, random_bots, 200))