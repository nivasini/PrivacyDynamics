import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Game:
	def __init__(self):
		pass
		# self.reward_matrix = reward_matrix
		# self.num_players = len(reward_matrix.ndim)
		# self.actions_per_player = reward_matrix.shape

	def rewards_from_actions(self, actions):
		return([0]) 

	def num_players(self):
		return(0)

	def num_interactions(self):
		return(0)

	def num_actions_per_interaction(self,ind):
		return(0)



class PriceDiscriminationGame(Game):
	def __init__(self,pr_high=0.5,v_high=15,v_low=5,cost=5):
		super().__init__()
		self.pr_high = pr_high
		self.v_high = v_high
		self.v_low = v_low
		self.cost_evade = cost

	# players are nature,buyer, seller
	def num_players(self):
		return(3)

	def reward_ind_buyer(self):
		return(1)
	def reward_ind_seller(self):
		return(2)

	# Interactions are:
	# (1) Type selection by nature
	# (2) Buyer signalling
	# (3) Seller setting price
	# (4) Buyer buying or not
	def num_interactions(self):
		return(4)
	def num_actions_per_interaction(self,ind):
		n_a = [2,2,2,2]
		return(n_a[ind])

	def is_high_value_buyer(self,actions):
		return(actions[0] == 0)
	def is_truthful_signalling(self,actions):
		return(actions[0] == actions[1])
	def is_low_price(self,actions):
		return(actions[2] == 0)
	def buyer_buys(self,actions):
		return(actions[3] == 0)
	def ind_high_val_sig(self):
		return(0)
	def ind_low_val_sig(self):
		return(1)
	def ind_high_val_buyer(self):
		return(0)
	def ind_low_val_buyer(self):
		return(1)
	def ind_buys(self):
		return(0)

	# Action sets per interaction
	# (1) 0: high value, 1: low value
	# (2) 0: high signal, 1: low signal
	# (3) 0: low price, 1: high price
	# (4) 0: buyer buys, 1: buyer does not buy
	def rewards_from_actions(self,actions):
		reward_nature = 0
		reward_non_active_buyer = 0
		high_value = self.is_high_value_buyer(actions)
		signal_truthful = self.is_truthful_signalling(actions)
		low_price = self.is_low_price(actions)
		buys = self.buyer_buys(actions)

		r_b = 0.0
		r_s = 0.0
		if (not signal_truthful):
			r_b -= self.cost_evade
			r_s -= self.cost_evade

		if(buys):
			if(low_price):
				r_s += self.v_low
				if(high_value):
					r_b += (self.v_high - self.v_low)
			else:
				r_s += self.v_high
				if(not high_value):
					r_b -= (self.v_high - self.v_low)

		rewards = []
		if(high_value):
			rewards = [reward_nature,r_b,r_s]
		else:
			rewards = [reward_nature,r_b,r_s]

		return(rewards)

	def eq_strat_high_value_buyer(self):
		if(self.cost_evade >= (self.v_high - self.v_low)):
			return(0)
		pr_high = self.pr_high
		v_low = self.v_low
		v_high = self.v_high
		return((1-pr_high)*v_low) / (pr_high * (v_high - v_low))

	def utility_without_pd_seller(self):
		utility_low_price = self.v_low
		utility_high_price = (self.pr_high * self.v_high)
		return(max(utility_low_price,utility_high_price))

	def utility_without_pd_buyer(self):
		price = self.v_low
		if(self.pr_high * self.v_high > self.v_low):
			price = self.v_high
		return(self.pr_high * (self.v_high - price))

	def eq_utility_seller(self):
		strat = self.eq_strat_high_value_buyer()
		pr_false_signal = self.pr_high * strat 
		u = -(pr_false_signal * self.cost_evade)
		pr_buy_low_price = (1 - self.pr_high) + (self.pr_high * strat)
		pr_buy_high_price = self.pr_high * (1-strat)
		u += ((pr_buy_high_price * self.v_high) + (pr_buy_low_price * self.v_low))
		return(u)

	def eq_utility_buyer(self):
		strat = self.eq_strat_high_value_buyer()
		u_lying = (self.v_high - self.v_low) - self.cost_evade
		return(self.pr_high * strat * u_lying)



class GameDynamics:
	def __init__(self,game,dynamics):
		self.game = game 
		self.num_players = game.num_players()
		self.num_interactions = game.num_interactions()
		self.dynamics = dynamics

	def run(self,T):
		rewards = np.zeros((self.num_players,T))
		print(rewards.shape)
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
		return(rewards)



class PlayerDynamic:
	def __init__(self,g,i,p):
		self.game = g 
		self.player_ind = p
		self.interaction_ind = i
		self.num_actions = g.num_actions_per_interaction(i)

	def reward_and_update(self,player_actions):
		return(self.game.rewards_from_actions(player_actions))

	def next_action(self,prev_player_actions):
		return(0)


class Exp3WithSignals(PlayerDynamic):
	def __init__(self,g,i,p,c=2):
		super().__init__(g,i,p)
		# c = self._num_prev_action_profiles()
		self.num_contexts = c
		self.num_arms = self.num_actions
		self.weights = [[1.0] * self.num_arms] * c 
		self.total_reward = [0.0] * c 
		self.t = [1] * c
		self.gamma_t = [0.1] * c
		self.use_context = True

	def next_action(self,prev_player_actions):
		c = prev_player_actions[1]
		if(not self.use_context):
			c = 0
		total_weight = sum(self.weights[c])
		self.gamma_t[c] = math.sqrt(math.log(self.num_arms))/self.t[c]
		probs = [(1-self.gamma_t[c])*weight/total_weight + self.gamma_t[c]/self.num_arms for weight in self.weights[c]]
		return (self._weighted_choice(probs))

	def reward_and_update(self, player_actions):
		p_ind = self.player_ind
		theta = player_actions[0]
		s = player_actions[1]
		c = s 
		if (not self.use_context):
			c = 0
		arm = player_actions[2]
		reward = self.game.rewards_from_actions(player_actions)[p_ind]
		self.total_reward[c] += reward
		self.t[c] += 1
		estimated_reward = reward / self.weights[c][arm]
		self.weights[c][arm] *= math.exp((1-self.gamma_t[c]) * estimated_reward / self.num_arms)
		return(reward)

	def _weighted_choice(self, weights,c=0):
		total_weight = sum(weights)
		rnd = total_weight * random.random()
		for i, weight in enumerate(weights):
		    rnd -= weight
		    if rnd < 0:
		        return i

    # def _num_prev_action_profiles(self):
    # 	n = 1
    # 	for i in range(p):
    # 		n *= super()/actions_per_player[i]
    # 	return(int(n))

class Exp3WithoutSignals(Exp3WithSignals):
	def __init__(self,g,i,p):
		super().__init__(g,i,p,1)
		self.use_context = False

	def next_action(self,prev_player_actions):
		a = []
		for p in prev_player_actions:
			a.append(p)
		a[1] = 0
		return(super().next_action(prev_player_actions))

	def reward_and_update(self,player_actions):
		return(super().reward_and_update(player_actions))

class randomStrategy(PlayerDynamic):
	def __init__(self,g,i,p,probs=[]):
		super().__init__(g,i,p)
		self.probs = probs

	def next_action(self,prev_player_actions):
		elements = range(self.num_actions)
		action = random.choices(elements,self.probs)[0]
		return(action)

class checkForSignalsUsage(PlayerDynamic):
	def __init__(self,g,i,p,tol=1):
		super().__init__(g,i,p)
		self.tol = tol 
		self.rewards_per_action = np.zeros(self.num_actions)
		self.num_each_action = np.zeros(self.num_actions)

	def next_action(self,prev_player_actions):
		low_sig = self.game.ind_low_val_sig()
		high_sig = self.game.ind_high_val_sig()
		is_high_val_buyer = self.game.is_high_value_buyer(prev_player_actions)
		if(not is_high_val_buyer):
			return(low_sig)
		return(high_sig)
		if(np.any(self.num_each_action == 0)):
			return(high_sig)
		avg_per_action = np.divide(self.rewards_per_action, self.num_each_action)
		if(np.any(np.abs(np.diff(avg_per_action))) > self.tol):
			elements = [low_sig,high_sig]
			eq_strat = self.game.eq_strat_high_value_buyer()
			choice = random.choices(elements,[eq_strat, 1.0 - eq_strat])
			return(int(choice))

		return(high_sig)

	def reward_and_update(self,player_actions):
		p_ind = self.player_ind
		int_ind = self.interaction_ind
		rewards = self.game.rewards_from_actions(player_actions)
		action = player_actions[int_ind]
		self.num_each_action[action] += 1
		self.rewards_per_action[action] += rewards[p_ind]
		return(rewards) 

class alwaysBuyWhenAffordable(PlayerDynamic):
	def __init__(self,g,i,p):
		super().__init__(g,i,p)

	def next_action(self,prev_player_actions):
		val = self.game.v_low
		if(self.game.is_high_value_buyer(prev_player_actions)):
			val = self.game.v_high
		price = self.game.v_high
		if(self.game.is_low_price(prev_player_actions)):
			price = self.game.v_low

		if(val < price): 
			return(1)
		return(0)











		