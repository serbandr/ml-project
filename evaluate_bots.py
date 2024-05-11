import pyspiel
import time


def evaluate_bots(state, bots, rng):
    """Plays bots against each other, returns terminal utility for each bot."""
    for bot in bots:
        bot.restart_at(state)
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choice(outcomes, p=probs)
            for bot in bots:
                bot.inform_action(state, pyspiel.PlayerId.CHANCE, action)
            state.apply_action(action)
        elif state.is_simultaneous_node():
            joint_actions = [
                bot.step(state)
                if state.legal_actions(player_id) else pyspiel.INVALID_ACTION
                for player_id, bot in enumerate(bots)
            ]
            state.apply_actions(joint_actions)
        else:
            current_player = state.current_player()
            # start = time.time()
            action = bots[current_player].step(state)
            # end = time.time()
            # print(f"Elapsed time for choosing action by{current_player}: {end - start} seconds")
            for i, bot in enumerate(bots):
                if i != current_player:
                    bot.inform_action(state, current_player, action)
            state.apply_action(action)
    print(state)
    return state.returns()
