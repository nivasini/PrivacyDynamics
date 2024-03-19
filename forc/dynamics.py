import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle
import pdb
import math

class Game:
    def __init__(self):
        return

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
    
    def num_estimators(self):
        return (1)

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
    def __init__(self, pr_high, v_high, v_low, cost, n, T):
        super().__init__()
        self.pr_high = pr_high
        self.v_high = v_high
        self.v_low = v_low
        self.cost_evade = cost
        self.n = n
        self.T = T

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
    def val_high_sig(self):
        return (0)

    def val_low_sig(self):
        return (1)

    def val_high_buyer(self):
        return (0)

    def val_low_buyer(self):
        return (1)

    def val_high_price(self):
        return (1)

    def val_low_price(self):
        return (0)

    def val_buy(self):
        return (0)
    
    def val_not_buy(self):
        return (1)

    # The following functions look at the action vector and return the truth value of
    # various attributes of the action vector
    def is_high_val_buyer(self, actions, buyer_idx):
        return (actions[0][buyer_idx] == 0)

    def is_truthful_signalling(self, actions, buyer_idx):
        return (actions[0][buyer_idx] == actions[1][buyer_idx])

    def is_low_price(self, actions, buyer_idx):
        return (actions[2][buyer_idx] == 0)
    
    def is_low_signal(self, actions, buyer_idx):
        return (actions[1][buyer_idx] == 1)

    def buyer_buys(self, actions, buyer_idx):
        return (actions[3][buyer_idx] == 0)

    # Returns rewards of each player given the action vectorx
    def rewards_from_actions(self, actions):
        num_buyers = len(actions[0])
        r_n = np.zeros(num_buyers)
        r_b = np.zeros(num_buyers)
        r_s = np.zeros(num_buyers)
        for j in range(num_buyers):
            high_value = self.is_high_val_buyer(actions, j)
            signal_truthful = self.is_truthful_signalling(actions, j)
            low_price = self.is_low_price(actions, j)
            buys = self.buyer_buys(actions, j)

            if (not signal_truthful):
                r_b[j] -= self.cost_evade
                r_s[j] -= self.cost_evade

            if (buys):
                if (low_price):
                    r_s[j] += self.v_low
                    if (high_value):
                        r_b[j] += (self.v_high - self.v_low)
                else:
                    r_s[j] += self.v_high
        
        r_n = np.mean(r_n)
        r_b = np.mean(r_b)
        r_s = np.mean(r_s)
        rewards = [r_n, r_b, r_s]
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

    # Returns the probability with which the high-value buyer signals
    # untruthfully in the SBPNE when alpha is greater than alpha*
    def q_star(self):
        return min(1, ((1 - self.pr_high) * self.v_low) / (self.pr_high * (self.v_high - self.v_low)))

    # Returns the probability with which the high-value buyer signals untruthfully in the SBPNE
    def prob_evade_high_val_buyer(self, alpha):
        if alpha <= self.cost_evade / (self.v_high - self.v_low):
            return (0)
        else:
            return self.q_star()

    # Returns buyer's equilibrium strategy in the one-stage game
    def buyer_eq_strat(self, buyer_val, alpha):
        low_buyer_val = self.val_low_buyer()
        high_buyer_val = self.val_high_buyer()

        if buyer_val == self.val_low_buyer():
            signal = buyer_val
        else:
            if alpha <= self.cost_evade / (self.v_high - self.v_low):
                signal = buyer_val
            else:
                signal = random.choices([low_buyer_val, buyer_val], [self.q_star(), 1 - self.q_star()])[0]
        return signal
    
    # Returns seller/buyer alpha-PD utilities
    def eq_utility(self, player_type, alpha):
        q = self.q_star()
        if player_type == 'seller':
            if self.v_low >= self.pr_high * self.v_high:
                if alpha <= self.cost_evade / (self.v_high - self.v_low):
                    u = self.v_low + (alpha * self.pr_high * (self.v_high - self.v_low))
                else:
                    u = self.v_low - (self.pr_high * self.cost_evade)
            else:
                if alpha <= self.cost_evade/(self.v_high - self.v_low):
                    u = (self.pr_high * self.v_high) + (alpha * (1 - self.pr_high) * self.v_low)
                else:
                    u = (self.pr_high * (self.v_high - self.cost_evade * q) + 
                         alpha * ((1 - self.pr_high) * self.v_low - 
                         self.pr_high * (self.v_high - self.v_low) * q))
        
        else:
            pr_high = 1 if player_type == 'high value buyer' else self.pr_high
            if self.v_low >= self.pr_high * self.v_high:
                if alpha <= self.cost_evade/(self.v_high - self.v_low):
                    u = pr_high * ((1 - alpha) * (self.v_high - self.v_low))
                else:
                    u = pr_high * ((self.v_high - self.v_low) - self.cost_evade)
            else:
                if alpha <= self.cost_evade/(self.v_high - self.v_low):
                    u = 0
                else:
                    u = pr_high * ((-self.cost_evade * q) + (alpha * (self.v_high - self.v_low) * q))
        return u

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
        self.T = self.game.T

    # Runs player dyamics for T rounds and returns rewards accumulated in each round
    def run(self, price_strat):
        rewards = np.zeros((self.num_players, self.T))
        actions = []
        alpha_hats = []
        alphas = []
        for t in range(self.T):
            if t % 10000 == 0:
                print('t',t)
            
            # players take an actions
            prev_player_actions = []
            for i in range(self.num_interactions):
                a = self.dynamics[i].next_action(prev_player_actions)
                prev_player_actions.append(a)
            actions.append(prev_player_actions)
            
            # compute rewards for nature, seller, and buyer
            r = self.game.rewards_from_actions(prev_player_actions)
            
            # update player parameters
            for i in range(self.num_interactions):
                updated_param = self.dynamics[i].update(prev_player_actions)
                if isinstance(self.dynamics[i], signalDynamic):
                    alpha_hats.append(updated_param)
                if isinstance(self.dynamics[i], priceDynamic):
                    if price_strat == 'Exp3':
                        alphas.append(0)
                    elif price_strat == '$\\alpha^*$-PD':
                        alphas.append(updated_param[0])
                    elif price_strat == 'CExp3':
                        alphas.append((updated_param[0][0]
                                       * updated_param[1][1])
                                       + (updated_param[0][1]
                                       * updated_param[1][0]))
            for p in range(self.num_players):
                rewards[p][t] = r[p]
        return rewards, actions, alpha_hats, alphas

# Captures the algorithm used for a given interaction in the game. 
# Attributes:
#	game: which game is being played
#	interaction_ind: Index of interaction this dynamic is defined for
#	player_ind: index of player taking action in this interaction
# Methods:
#	next_action: based on the history (games played up to this point) and the actions in the current
#					game from previous interactions, the action the player takes in the current interaction
#	update: takes player_actions and updates the history and hence the states of the dynamics
class PlayerDynamic:
    def __init__(self, g, i, p, player_args):
        self.game = g
        self.interaction_ind = i
        self.player_ind = p
        self.num_actions = g.num_actions_per_interaction(i)
        self.player_args = player_args

    def update(self, player_actions):
        return

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
                actions_bl[self.interaction_ind] = [a for _ in range(self.game.n)]
                rewards_baseline[a][t] = self.game.rewards_from_actions(actions_bl)[self.player_ind]

        avg_cum_rewards = cumulative_average(rewards.tolist())
        avg_cum_baseline_rewards = []
        for a in range(self.num_actions):
            avg_cum_baseline_rewards.append(cumulative_average(rewards_baseline[a].tolist()))

        avg_cum_regret = np.zeros(T)
        for t in range(T):
            m = -np.inf
            for a in range(self.num_actions):
                if (avg_cum_baseline_rewards[a][t] > m):
                    m = avg_cum_baseline_rewards[a][t]
            avg_cum_regret[t] = max(0, m - avg_cum_rewards[t])

        return (avg_cum_regret)

'''
Player Strategies
'''
class ContextualDynamic(PlayerDynamic):
    def __init__(self, g, i, p, strategy_name, num_dynamics, player_args, contexts):
        super().__init__(g, i, p, player_args)
        self.num_dynamics = num_dynamics
        self.context_inds = contexts['inds']
        self._initialize_dynamics(strategy_name)
        self.player_args = player_args
        self.assignments = None
        if contexts['function']:
            self.contextual_function = eval(contexts['function'])

    def _initialize_dynamics(self, strategy_name):
        class_name = globals()[strategy_name]
        args = [self.game, self.interaction_ind, self.player_ind, self.player_args]
        strategy_dynamics = []
        for i in range(self.num_dynamics):
            strategy_dynamics.append(class_name(*args))
        self.dynamics = strategy_dynamics

    def _assign_estimators(self, prev_player_actions):
        num_estimators = self.player_args['num_estimators']
        pr_estimators = self.player_args['prob_estimators']
        assignments = [[] for _ in range(num_estimators)]
        for j in range(self.game.n):
            choice = random.choices(range(num_estimators), pr_estimators)[0]
            assignments[choice].append(j)
        return assignments

    def _assign_dynamics(self, prev_player_actions):
        context_actions = [prev_player_actions[i] for i in self.context_inds][0]
        unique_actions = self.game.num_actions_per_interaction(self.interaction_ind) 
        assignments = [[] for _ in range(unique_actions)]
        for action in range(unique_actions):
            dynamic_idxs = list(np.where(np.array(context_actions)==action)[0])
            assignments[action] = dynamic_idxs
        return assignments

    def _make_contextual_assignments(self, prev_player_actions):
        assignments = self.contextual_function(prev_player_actions)
        self.assignments = assignments

    def next_action(self, prev_player_actions):
        if self.num_dynamics==1:
            self.assignments = [list(np.arange(self.game.n))]
            actions = self.dynamics[0].next_action(prev_player_actions)
        else:
            actions = np.zeros(self.game.n).astype(np.int32)
            self._make_contextual_assignments(prev_player_actions)
            for i in range(self.num_dynamics):
                if not self.assignments[i]:
                    continue
                prev_actions_per_dynamic = list(map(lambda x:
                                                    list(np.array(x)[self.assignments[i]]),
                                                    prev_player_actions))
                actions_per_dynamic = self.dynamics[i].next_action(prev_actions_per_dynamic)
                actions[self.assignments[i]] = actions_per_dynamic
        return actions

    def update(self, player_actions):
        updated_params = []
        for i in range(self.num_dynamics):
            actions_per_dynamic = list(map(lambda x: list(np.array(x)[self.assignments[i]]), player_actions))
            updated_param = self.dynamics[i].update(actions_per_dynamic)
            updated_params.append(updated_param)
        return updated_params

'''
Nature's Strategy
'''

# Strategy that randomly selects buyer's type
class randomSelection(PlayerDynamic):
    def __init__(self, g, i, p, player_args):
        super().__init__(g, i, p, player_args)

    def next_action(self, prev_player_actions):
        high_buyer = self.game.val_high_buyer()
        low_buyer = self.game.val_low_buyer()
        actions = []
        for j in range(self.game.n):
            action = random.choices([low_buyer, high_buyer], [1 - self.game.pr_high, self.game.pr_high])[0]
            actions.append(action)
        return actions

    def update(self, prev_player_actions):
        return

class natureDynamic(ContextualDynamic):
    def __init__(self, g, i, p, strategy_name, num_dynamics, player_args, contexts):
        super().__init__(g, i, p, strategy_name, num_dynamics, player_args, contexts)

'''
Signal Strategies
'''
# Buyer's signaling strategy with a consistent estimator of seller's probability of price discrimination at each round
class consistentEstimate(PlayerDynamic):
    def __init__(self, g, i, p, player_args):
        super().__init__(g, i, p, player_args)
        self.price_per_action = np.zeros(self.num_actions)
        self.num_each_action = np.zeros(self.num_actions)
        self.num_rounds = 1
        self.alpha_hat = 0.5
        self.normalized_num_rounds_pd = 0
        self.player_args = player_args
        self.pr_flip = self.player_args['pr_flip']
        self.noise_bounds = self.player_args['noise_bounds']
        self.noise = 0.0
        self.round = 1
    
    def flip_action(self, action):
        opposite_action = self.game.val_high_sig() if action==self.game.val_low_sig() else self.game.val_low_sig()
        choice = random.choices([action, opposite_action], [1 - self.pr_flip, self.pr_flip])[0]
        return choice

    def next_action(self, prev_player_actions):
        buyer_vals = prev_player_actions[self.interaction_ind - 1]
        num_buyers = len(buyer_vals)
        actions = []
        for j in range(num_buyers):
            buyer_val = buyer_vals[j]
            action = self.game.buyer_eq_strat(buyer_val, self.alpha_hat)
            if buyer_val == self.game.val_high_buyer():
                action = self.flip_action(action)
            actions.append(action)
        return actions

    def update(self, player_actions):
        
        low_sig = self.game.val_low_sig()
        high_sig = self.game.val_high_sig()
        low_price = self.game.val_low_price()
        high_price = self.game.val_high_price()

        signal_ind = self.interaction_ind
        price_ind = self.interaction_ind + 1
        
        self.num_rounds += 1
        round_is_inf = 1 if (0 in player_actions[signal_ind] and 1 in player_actions[signal_ind]) else 0
        if round_is_inf:
            price_for_low_sig = player_actions[price_ind][player_actions[signal_ind].index(low_sig)]
            price_for_high_sig = player_actions[price_ind][player_actions[signal_ind].index(high_sig)]
            pd_detected = 1 if price_for_low_sig != price_for_high_sig else 0
            
            prob_all_low_signals = (((1 - self.game.pr_high) +  
                                      self.game.pr_high
                                     * self.game.prob_evade_high_val_buyer(self.alpha_hat)) ** self.game.n)
            prob_all_high_signals = (self.game.pr_high * (1 - self.game.prob_evade_high_val_buyer(self.alpha_hat))) ** self.game.n
            prob_round_is_inf = 1 - (prob_all_low_signals + prob_all_high_signals)
            self.normalized_num_rounds_pd += ((pd_detected * round_is_inf) / prob_round_is_inf) 
        true_alpha_hat = self.normalized_num_rounds_pd / self.num_rounds
        noise = np.random.uniform(*self.noise_bounds)
        self.alpha_hat = min(1, max(true_alpha_hat + noise, 0))
        return self.alpha_hat

class signalDynamic(ContextualDynamic):
    def __init__(self, g, i, p, strategy_name, num_dynamics, player_args, contexts):
        super().__init__(g, i, p, strategy_name, num_dynamics, player_args, contexts)

'''
Price Strategies
'''

# alpha-PD strategy
class alphaPD(PlayerDynamic):
    def __init__(self, g, i, p, player_args):
        super().__init__(g, i, p, player_args)
        self.player_args = player_args
        self.alpha = self.player_args['alpha']

    def next_action(self, prev_player_actions):
        low_price = self.game.val_low_price()
        high_price = self.game.val_high_price()
        pd = random.choices([0, 1], [1 - self.alpha, self.alpha])[0]
        if pd==1:
            actions = [low_price if self.game.is_low_signal(prev_player_actions, j) else high_price for j in range(self.game.n)]
        else:
            actions = [low_price if self.game.v_low >= self.game.pr_high * self.game.v_high else high_price for _ in range(self.game.n)]
        return actions
    
    def update(self, prev_player_actions):
        return self.alpha

# Implemention of EXP3-IX from Chapter 12 of Bandits book by Szepesvari and Lattimore
class Exp3(PlayerDynamic):
    def __init__(self, g, i, p, player_args):
        super().__init__(g, i, p, player_args)
        self.num_arms = self.num_actions
        self.weights = [1.0] * self.num_arms
        self.t = 1
        self.eta = np.sqrt(np.log(self.num_arms) / (self.game.T * self.num_arms))
        self.gamma = 0
        self.probabilities = self.get_probabilities()

    def get_probabilities(self):
        total_weight = np.sum(self.weights)
        probs = self.weights / total_weight
        return probs

    def next_action(self, prev_player_actions):
        probs = self.get_probabilities()
        # this ensures buyer utility converges to alpha=1 in CExp3
        #if 0 not in prev_player_actions[1]:
        #    probs = [1, 0]
        action = random.choices(range(self.num_arms), probs)[0]
        num_buyers = np.shape(prev_player_actions)[1]
        return [action] * num_buyers

    def normalized_reward(self, r):
        max_rewards = self.game.max_rewards()
        min_rewards = self.game.min_rewards()
        l = min_rewards[self.player_ind]
        h = max_rewards[self.player_ind]
        return (r - l) / (h - l)

    def update(self, player_actions):
        if not any(player_actions):
            reward = None
        else:
            p_ind = self.player_ind
            arm = player_actions[self.interaction_ind][0]
            r = self.game.rewards_from_actions(player_actions)[p_ind]
            reward = self.normalized_reward(r)
            loss = 1.0 - reward
            prob = self.get_probabilities()[arm]
            estimated_loss = loss / (prob + self.gamma)
            self.weights[arm] *= math.exp(-self.eta * estimated_loss)
            self.t += 1
        return self.get_probabilities()

class priceDynamic(ContextualDynamic):
    def __init__(self, g, i, p, strategy_name, num_dynamics, player_args, contexts):
        super().__init__(g, i, p, strategy_name, num_dynamics, player_args, contexts)

'''
Buy Strategy
'''
# Buyer's buying strategy -- buyer buys when value is at least price
class buyUnderValue(PlayerDynamic):
    def __init__(self, g, i, p, player_args):
        super().__init__(g, i, p, player_args)

    def next_action(self, prev_player_actions):
        prices = prev_player_actions[self.interaction_ind - 1]
        low_price = self.game.val_low_price()
        high_price = self.game.val_high_price()
        buy = self.game.val_buy()
        not_buy = self.game.val_not_buy()

        actions = []
        for j in range(self.game.n):
            if prices[j] == low_price:
                action = buy
            else:
                action = buy if self.game.is_high_val_buyer(prev_player_actions, j) else not_buy
            actions.append(action)
        return actions

    def update(self, prev_player_actions):
        return

class buyDynamic(ContextualDynamic):
    def __init__(self, g, i, p, strategy_name, num_dynamics, player_args, contexts):
        super().__init__(g, i, p, strategy_name, num_dynamics, player_args, contexts)

'''
Helper Functions
'''
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
