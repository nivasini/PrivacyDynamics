import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pulp
import copy


class Game:
    def __init__(self):
        pass

    def rewards_from_actions(self, actions):
        return ([0])

    def num_players(self):
        return (0)

    def num_interactions(self):
        return (0)

    def num_actions_per_interaction(self, ind):
        return (0)

    def max_rewards(self):
        return ([0])

    def min_rewards(self):
        return ([0])


# The price discrimination game: this is a game between a buyer and a seller 
# There are two types of buyers. The buyer's type is randomly picked at the beginning by nature
# The type indicates the value of the buyer. This can take either v_low or v_high with v_low < v_high
# The buyer moves first and signals that their value is either high or low. 
# To signal differently than the true value, the buyer needs to pay a cost cost_evade. 
# The seller also suffers a cost of cost_evade for untruthful signalling.
# Based on the signal, the seller sets a price for the item.
# The buyer either buys or does not buy the item. 
# The seller gets reward of the price set if the buyer buys and zero otherwise.
# The buyer gets reward value minus price if they buy and zero otherwise. 
class PriceDiscriminationGame(Game):
    def __init__(self, pr_high=0.5, v_high=15, v_low=5, cost=5):
        super().__init__()
        self.pr_high = pr_high
        self.v_high = v_high
        self.v_low = v_low
        self.cost_evade = cost

    # players are nature,buyer, seller
    def num_players(self):
        return (3)

    # The index in the reward vector representing the buyer's reward
    def reward_ind_buyer(self):
        return (1)

    # The index in the reward vector representing the seller's reward
    def reward_ind_seller(self):
        return (2)

    # Returns number of interactions in each game
    # Interactions are:
    # (1) Type selection by nature
    # (2) Buyer signalling
    # (3) Seller setting price
    # (4) Buyer buying or not
    def num_interactions(self):
        return (4)

    # Returns the number of actions in interaction with index ind
    def num_actions_per_interaction(self, ind):
        n_a = [2, 2, 2, 2]
        return (n_a[ind])

    # The action vector: vector indicating action at each interaction
    # What action each value of the action vector indicates:
    # (1) 0: high value, 1: low value
    # (2) 0: high signal, 1: low signal
    # (3) 0: low price, 1: high price
    # (4) 0: buyer buys, 1: buyer does not buy

    # The following functions denote the value the action is set to in the action vector
    def val_high_val_sig(self):
        return (0)

    def val_low_val_sig(self):
        return (1)

    def val_high_val_buyer(self):
        return (0)

    def val_low_val_buyer(self):
        return (1)

    def val_buys(self):
        return (0)

    # The following functions look at the action vector and return the truth value of
    # various attributes of the action vector
    def is_high_value_buyer(self, actions):
        return (actions[0] == 0)

    def is_truthful_signalling(self, actions):
        return (actions[0] == actions[1])

    def is_low_price(self, actions):
        return (actions[2] == 0)

    def buyer_buys(self, actions):
        return (actions[3] == 0)

    # Returns rewards of each player given the action vector
    def rewards_from_actions(self, actions):
        reward_nature = 0
        high_value = self.is_high_value_buyer(actions)
        signal_truthful = self.is_truthful_signalling(actions)
        low_price = self.is_low_price(actions)
        buys = self.buyer_buys(actions)

        r_b = 0.0
        r_s = 0.0
        if (not signal_truthful):
            r_b -= self.cost_evade
            r_s -= self.cost_evade

        if (buys):
            if (low_price):
                r_s += self.v_low
                if (high_value):
                    r_b += (self.v_high - self.v_low)
            else:
                r_s += self.v_high
                if (not high_value):
                    r_b -= (self.v_high - self.v_low)

        rewards = []
        if (high_value):
            rewards = [reward_nature, r_b, r_s]
        else:
            rewards = [reward_nature, r_b, r_s]

        return (rewards)

    # Returns maximum reward each player can get
    def max_rewards(self):
        s = self.v_high
        b = self.v_high - self.v_low
        return ([0, b, s])

    # Returs minimum reward each player can get
    def min_rewards(self):
        s = 0.0 - self.cost_evade
        b = 0.0 - self.cost_evade
        return ([0, b, s])

    # The Subgame Perfect Bayes Nash Equilibrium (SBPNE):
    # In the unique SBPNE, the buyer with value v_low always signals truthfully
    # The buyer with v_high value signals untruthfully with a certain probability.
    # The seller sets price v_high when seeing the high value signal
    # The seller sets price v_low when seeing the low value signal

    # Returns the probability with which buyer with value v_high signals untruthfully in the SBPNE
    def eq_strat_high_value_buyer(self):
        if (self.cost_evade >= (self.v_high - self.v_low)):
            return (0)
        pr_high = self.pr_high
        v_low = self.v_low
        v_high = self.v_high
        return ((1 - pr_high) * v_low) / (pr_high * (v_high - v_low))

    # Returns the utility of the seller in the SBPNE
    def eq_utility_seller(self):
        strat = self.eq_strat_high_value_buyer()
        pr_false_signal = self.pr_high * strat
        u = -(pr_false_signal * self.cost_evade)
        pr_buy_low_price = (1 - self.pr_high) + (self.pr_high * strat)
        pr_buy_high_price = self.pr_high * (1 - strat)
        u += ((pr_buy_high_price * self.v_high) + (pr_buy_low_price * self.v_low))
        return (u)

    # Returns the utility of the buyer in the SBPNE
    def eq_utility_buyer(self):
        strat = self.eq_strat_high_value_buyer()
        u_lying = (self.v_high - self.v_low) - self.cost_evade
        return (self.pr_high * strat * u_lying)

    # Game without price discrimination: consisits of the same interactions, except that
    # the seller cannot set the price based on the signal.
    # So the buyer never signals untruthfully when there is no price discrimination.

    # Returns the seller's utility when there is no price discrimination
    def utility_without_pd_seller(self):
        utility_low_price = self.v_low
        utility_high_price = (self.pr_high * self.v_high)
        return (max(utility_low_price, utility_high_price))

    # Returns the buyer's utility when there is no price discrimination
    def utility_without_pd_buyer(self):
        price = self.v_low
        if (self.pr_high * self.v_high > self.v_low):
            price = self.v_high
        return (self.pr_high * (self.v_high - price))

    # Returns a CCE of the game
    def compute_CCE(self):
        problem = pulp.LpProblem("LP Problem", pulp.LpMinimize)
        p00 = pulp.LpVariable('p00', lowBound=0)
        p01 = pulp.LpVariable('p01', lowBound=0)
        p10 = pulp.LpVariable('p10', lowBound=0)
        p11 = pulp.LpVariable('p11', lowBound=0)
        objective = p00 + p01
        problem += objective
        constr0 = p00 + p01 + p10 + p11 <= 1
        constr1 = p00 + p01 + p10 + p11 >= 1
        constr2 = p00 * (self.pr_high * self.v_high - self.v_low) <= p10 * (1.0 - self.pr_high) * self.v_low
        constr3 = p11 * (1.0 - self.pr_high) * self.v_low <= p01 * ((self.pr_high * self.v_high) - self.v_low)
        constr4 = p10 * (self.v_high - self.v_low - self.cost_evade) <= p11 * self.cost_evade
        constr5 = p01 * self.cost_evade <= p00 * (self.v_high - self.v_low - self.cost_evade)
        problem += constr0
        problem += constr1
        problem += constr2
        problem += constr3
        problem += constr4
        problem += constr5
        problem.solve()
        p00_val = p00.value()
        p01_val = p01.value()
        p10_val = p10.value()
        p11_val = p11.value()
        return ([p00_val, p01_val, p10_val, p11_val])


# Returns the cumulative average of a list
def cumulative_average(l):
    cumulative_sum = 0
    cumulative_count = 0
    cumulative_average = []

    for num in l:
        cumulative_sum += num
        cumulative_count += 1
        cumulative_average.append(cumulative_sum / cumulative_count)

    return (cumulative_average)


# When a game is played repeatedly, the dynamics are captured by the following Dynamics classes

# Encapsulates the algorithm used in each interaction of the game 
# Has attributes game: denotes which game is being played, 
# dynamics: a list with objects of PlayerDynamic type denoting the dynamic employed in each round
class GameDynamics:
    def __init__(self, game, dynamics):
        self.game = game
        self.num_players = game.num_players()
        self.num_interactions = game.num_interactions()
        self.dynamics = dynamics

    # Runs player dyamics for T rounds and returns rewards accumulated in each round
    def run(self, T):
        rewards = np.zeros((self.num_players, T))
        actions = []
        for t in range(T):
            prev_player_actions = []
            for i in range(self.num_interactions):
                a = self.dynamics[i].next_action(prev_player_actions)
                prev_player_actions.append(a)
            r = self.game.rewards_from_actions(prev_player_actions)
            for i in range(self.num_interactions):
                x = self.dynamics[i].reward_and_update(prev_player_actions)
            for p in range(self.num_players):
                rewards[p][t] = r[p]
            actions.append(prev_player_actions)
        return (actions, rewards)


# Captures the algorithm used for a given interaction in the game. 
# Attributes:
#	game: which game is being played
#	interaction_ind: Index of interaction this dynamic is defined for
#	player_ind: index of player taking action in this interaction
# Methods:
#	next_action: based on the history (games played up to this point) and the actions in the current
#					game from previous interactions, the action the player takes in the current interaction
#	reward_and_update: takes player_actions and updates the history and hence the states of the dynamics
class PlayerDynamic:
    def __init__(self, g, i, p):
        self.game = g
        self.player_ind = p
        self.interaction_ind = i
        self.num_actions = g.num_actions_per_interaction(i)

    def reward_and_update(self, player_actions):
        return (self.game.rewards_from_actions(player_actions))

    def next_action(self, prev_player_actions):
        return (0)

    def avg_regret(self, actions):
        T = len(actions)
        rewards_baseline = np.zeros((self.num_actions, T))
        rewards = np.zeros(T)
        actions_for_baseline = copy.deepcopy(actions)
        for t in range(T):
            rewards[t] = self.game.rewards_from_actions(actions[t])[self.player_ind]
            actions_bl = actions_for_baseline[t]
            for a in range(self.num_actions):
                actions_bl[self.interaction_ind] = a
                rewards_baseline[a][t] = self.game.rewards_from_actions(actions_bl)[self.player_ind]

        avg_cum_rewards = cumulative_average(rewards.tolist())
        avg_cum_baseline_rewards = []
        for a in range(self.num_actions):
            avg_cum_baseline_rewards.append(cumulative_average(rewards_baseline[a].tolist()))

        avg_cum_regret = np.zeros(T)
        for t in range(T):
            m = -np.inf
            for a in range(self.num_arms):
                if (avg_cum_baseline_rewards[a][t] > m):
                    m = avg_cum_baseline_rewards[a][t]
            avg_cum_regret[t] = max(0, m - avg_cum_rewards[t])

        return (avg_cum_regret)


# Implementation of EXP3_S algorithm from the paper 'Nonstochastic MAB problem' - Auer et al.
# Guaranteed to have worst case sublinear regret against adversarial sequences
class EXP3_S(PlayerDynamic):
    def __init__(self, g, i, p):
        super().__init__(g, i, p)
        self.num_arms = self.num_actions
        self.weights = np.ones(self.num_arms)
        self.time_step = 1
        self.eta = np.sqrt(np.log(self.num_arms) / (self.num_arms * 2))
        self.current_horizon = 2

    def get_probabilities(self):
        total_weight = np.sum(self.weights)
        probs = self.weights / total_weight
        return probs

    def next_action(self, prev_player_actions):
        probs = self.get_probabilities()
        return np.random.choice(self.num_arms, p=probs)

    # Normalize rewards to lie in [0,1]
    def normalized_reward(self, r):
        max_rewards = self.game.max_rewards()
        min_rewards = self.game.min_rewards()
        l = min_rewards[self.player_ind]
        h = max_rewards[self.player_ind]
        return ((r - l) / (h - l))

    def reward_and_update(self, player_actions):
        probs = self.get_probabilities()
        arm = player_actions[self.interaction_ind]
        r = self.game.rewards_from_actions(player_actions)[self.player_ind]
        reward = self.normalized_reward(r)
        estimated_reward = reward / probs[arm]

        self.weights[arm] *= np.exp(self.eta * estimated_reward)

        self.time_step += 1
        if self.time_step > self.current_horizon:
            # If we've passed the current horizon, reset the weights and double the horizon
            self.weights = np.ones(self.num_arms)
            self.current_horizon *= 2
            self.eta = np.sqrt(np.log(self.num_arms) / (self.num_arms * 2 * self.current_horizon))


class Exp3(PlayerDynamic):
    def __init__(self, g, i, p):
        super().__init__(g, i, p)
        self.num_arms = self.num_actions
        self.weights = [1.0] * self.num_arms
        self.total_reward = 0.0
        self.t = 1
        self.gamma_t = 0.1

    def get_probabilities(self):
        total_weight = np.sum(self.weights)
        probs = (1 - self.gamma_t) * (self.weights / total_weight) + self.gamma_t / self.num_arms
        return probs

    def next_action(self, prev_player_actions):
        self.gamma_t = math.sqrt(math.log(self.num_arms)) / self.t
        probs = self.get_probabilities()
        return (self._weighted_choice(probs))

    def normalized_reward(self, r):
        max_rewards = self.game.max_rewards()
        min_rewards = self.game.min_rewards()
        l = min_rewards[self.player_ind]
        h = max_rewards[self.player_ind]
        return ((r - l) / (h - l))

    def reward_and_update(self, player_actions):
        p_ind = self.player_ind
        arm = player_actions[self.interaction_ind]
        r = self.game.rewards_from_actions(player_actions)[p_ind]
        reward = self.normalized_reward(r)
        self.total_reward += reward
        self.t += 1
        if (self.weights[arm] > 0 or True):
            estimated_reward = reward / self.weights[arm]
            self.weights[arm] *= math.exp((1 - self.gamma_t) * estimated_reward / self.num_arms)
        return (reward)

    def _weighted_choice(self, probs, c=0):
        total_weight = sum(probs)
        rnd = total_weight * random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd < 0:
                return i


class ContextualDynamic(PlayerDynamic):
    def __init__(self, g, i, p, c_inds, dynamic_name):
        super().__init__(g, i, p)
        self.context_inds = c_inds
        self.num_contexts = self._num_contexts()
        self._initialize_dynamic_per_context(dynamic_name)

    def _num_contexts(self):
        n = 1
        for i in self.context_inds:
            num_actions = self.game.num_actions_per_interaction(i)
            n *= num_actions
        return (n)

    def _initialize_dynamic_per_context(self, dynamic_name):
        contexts_shape = [self.game.num_actions_per_interaction(i) for i in self.context_inds]
        class_name = globals()[dynamic_name]
        args = [self.game, self.interaction_ind, self.player_ind]

        def create_array(depth):
            if depth == len(contexts_shape):
                return (class_name(*args))
            arr = np.empty(contexts_shape[depth], dtype=object)

            for i in range(contexts_shape[depth]):
                arr[i] = create_array(depth + 1)

            return (arr)

        self.dynamics = create_array(0)

    def _dynamic_from_player_actions(self, a):
        contexts = [a[i] for i in self.context_inds]
        d = self.dynamics[tuple(contexts)]
        return (d)

    def next_action(self, prev_player_actions):
        d = self._dynamic_from_player_actions(prev_player_actions)
        return (d.next_action(prev_player_actions))

    def reward_and_update(self, player_actions):
        d = self._dynamic_from_player_actions(player_actions)
        return (d.reward_and_update(player_actions))


def base_n_to_decimal(number_list, n):
    decimal_number = 0
    power = 0

    for digit in reversed(number_list):
        decimal_number += digit * (digit ** power)
        power += 1

    return (decimal_number)


def decimal_to_base_n(decimal_number, n):
    if decimal_number == 0:
        return ([0])

    result = []
    while (decimal_number > 0):
        remainder = decimal_number % n
        result.insert(0, remainder)
        decimal_number //= n

    return (result)


# create a modified game where the actions are mappings
# from contexts to the actions of the original game g
# contexts_per_player[i] is the list of indices of actions
# the player making action i has access to when taking action
# the new action this player takes is a mapping from realizations
# of contexts_per_player[i] to an action
class GameWithContextualActions(Game):
    def __init__(self, g, contexts_per_interaction):
        super().__init__()
        self.contexts_per_interaction = contexts_per_interaction
        self.base_game = g

    def num_players(self):
        return self.base_game.num_players()

    def num_interactions(self):
        return self.base_game.num_interactions()

    def max_rewards(self):
        return self.base_game.max_rewards()

    def min_rewards(self):
        return self.base_game.min_rewards()

    def num_context_realizations(self, ind):
        contexts = self.contexts_per_interaction[ind]
        n = 1
        for c in contexts:
            n *= self.base_game.num_actions_per_interaction(c)
        return int(n)

    def num_actions_per_interaction(self, ind):
        a = self.base_game.num_actions_per_interaction(ind)
        n = self.num_context_realizations(ind)
        return pow(a, n)

    def _context_vals_to_ind(self, c_inds, c_vals):
        if len(c_inds) == 0:
            return 0
        r = 1
        p = 1
        for i in range(len(c_inds)):
            ind = c_inds[i]
            n = self.contexts_per_interaction[ind]
            v = c_vals[i]
            r += (p * v)
            p *= n

        return int(r)

    def _ind_to_context_vals(self, ind, c_inds):
        r = []
        p = 1
        a = ind
        for c in c_inds:
            n = self.contexts_per_interaction[c]
            v = a % n
            a //= n
            r.append(v)

        return r

    # Take action in this game and it's interaction index and maps to the corresponding mapping in the base game
    # The mapping in the base game is a mapping from values of the context indices to a an action in the base game
    # at the interaction index
    def action_to_base_game_mapping(self, action, int_ind, context_vals):
        num_actions = self.num_actions_per_interaction(int_ind)
        num_contexts = self.num_context_realizations(int_ind)
        contexts = self.contexts_per_interaction[int_ind]
        action_vals = decimal_to_base_n(action, num_actions)
        # since action_vals is the mapping applied to all possible contexts,
        # it needs to be of length num_contexts.
        # so we will pad the front of action_vals with sufficient zeros
        l = len(action_vals)
        for i in range(num_contexts - l):
            action_vals.insert(0, 0)
        context_vals_ind = self._context_vals_to_ind(contexts, context_vals)
        realized_action = action_vals[context_vals_ind]
        return realized_action

    # given the action index action in the new game
    # and the values of the context inds c_vals,
    # finds the action in the base game
    def actions_base_game(self, action_profile):
        num_ints = len(action_profile)
        realized_action_profile = []
        for i in range(num_ints):
            a = action_profile[i]
            contexts = self.contexts_per_interaction[i]
            context_vals = []
            for c in contexts:
                context_vals.append(action_profile[c])

            realized_action = self.action_to_base_game_mapping(a, i, context_vals)
            realized_action_profile.append(realized_action)
        return realized_action_profile

    def rewards_from_actions(self, actions):
        realized_actions = self.actions_base_game(actions)
        rewards = self.base_game.rewards_from_actions(realized_actions)
        return rewards

class ContextualExp3(ContextualDynamic):
    def __init__(self, g, i, p, c_inds):
        super().__init__(g, i, p, c_inds, 'EXP3_S')


# Strategy that randomly picks an action according to the attribute probs
class randomStrategy(PlayerDynamic):
    def __init__(self, g, i, p, probs=[]):
        super().__init__(g, i, p)
        self.probs = probs

    def next_action(self, prev_player_actions):
        elements = range(self.num_actions)
        action = random.choices(elements, self.probs)[0]
        return (action)


# Dynamic for the buyer in the price-discrimination game where the buyer checks if the seller uses the signal to price discriminate.
# Detect price discrimination if the difference in average price for the different signals are greater than the attribute tol
# If no price discrimination determined, the buyer signals truthfully
# If price discrimination determined, the buyer plays the SPBNE strategy
class checkForSignalsUsage(PlayerDynamic):
    def __init__(self, g, i, p, tol=1, is_contextual=False):
        super().__init__(g, i, p)
        self.tol = tol
        self.price_per_action = np.zeros(self.num_actions)
        self.num_each_action = np.zeros(self.num_actions)
        self.contextual = is_contextual
        self.contextual_game = None
        if is_contextual:
            self.game = g.base_game
            self.contextual_game = g

    def next_action(self, prev_player_actions_arg):
        prev_player_actions = copy.deepcopy(prev_player_actions_arg)
        if self.contextual:
            prev_player_actions = self.contextual_game.actions_base_game(prev_player_actions)
        low_sig = self.game.val_low_val_sig()
        high_sig = self.game.val_high_val_sig()
        is_high_val_buyer = self.game.is_high_value_buyer(prev_player_actions)
        if not is_high_val_buyer:
            return low_sig
        # return(high_sig)
        if np.any(self.num_each_action == 0):
            return high_sig
        avg_per_action = np.divide(self.price_per_action, self.num_each_action)
        if np.any(np.abs(np.diff(avg_per_action)) > self.tol):
            elements = [low_sig, high_sig]
            eq_strat = self.game.eq_strat_high_value_buyer()
            choice = random.choices(elements, [eq_strat, 1.0 - eq_strat])
            return int(choice[0])

        return high_sig

    def reward_and_update(self, player_actions_arg):
        player_actions = copy.deepcopy(player_actions_arg)
        if self.contextual:
            player_actions = self.contextual_game.actions_base_game(player_actions)
        int_ind = self.interaction_ind
        rewards = self.game.rewards_from_actions(player_actions)
        action = player_actions[int_ind]
        price = self.game.v_high
        if self.game.is_low_price(player_actions):
            price = self.game.v_low
        self.num_each_action[action] += 1
        self.price_per_action[action] += price
        return rewards


# Strategy for deciding to buy where the buyer always buys when the price is not greater than the buyer's value
class alwaysBuyWhenAffordable(PlayerDynamic):
    def __init__(self, g, i, p, is_contextual=False):
        super().__init__(g, i, p)
        self.contextual = is_contextual
        self.contextual_game = None
        self.is_contextual = is_contextual
        if is_contextual:
            self.game = g.base_game
            self.contextual_game = g

    def next_action(self, prev_player_actions):
        val = self.game.v_low
        if self.is_contextual:
            prev_player_actions = self.contextual_game.actions_base_game(prev_player_actions)
        if self.game.is_high_value_buyer(prev_player_actions):
            val = self.game.v_high
        price = self.game.v_high
        if self.game.is_low_price(prev_player_actions):
            price = self.game.v_low

        if val < price:
            return 1
        return 0
