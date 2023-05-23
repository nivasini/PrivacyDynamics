import math
import random
import matplotlib.pyplot as plt
import numpy as np
from dynamics import *
from matplotlib import gridspec

# Returns the cumulative average of a list
def cumulative_average(l):
	cumulative_sum = 0
	cumulative_count = 0
	cumulative_average = []

	for num in l:
	    cumulative_sum += num
	    cumulative_count += 1
	    cumulative_average.append(cumulative_sum / cumulative_count)

	return(cumulative_average)

def create_price_disc_instance(pr_high,v_high,v_low,cost_evade, dynamics_names,extra_args):
	price_disc_game = PriceDiscriminationGame(pr_high,v_high,v_low,cost_evade)
	ind_nature = 0
	ind_buyer = price_disc_game.reward_ind_buyer()
	ind_seller = price_disc_game.reward_ind_seller()
	num_interactions = price_disc_game.num_interactions()
	inds = [ind_nature,ind_buyer,ind_seller,ind_buyer]

	dynamics = []
	for i in range(num_interactions):
		class_name = globals()[dynamics_names[i]]
		ind = inds[i]
		arg = [price_disc_game,i,ind] + extra_args[i]
		dynamics.append(class_name(*arg))

	return(GameDynamics(price_disc_game,dynamics))

def plot_cum_buyer_seller_utilities(r, d = None):
	num_runs = len(r)
	s = []
	b = []
	for i in range(num_runs):
		b.append(cumulative_average(r[i][1]))
		s.append(cumulative_average(r[i][2]))
	hlines = None
	if d is not None:
		rows = 2
		cols = num_runs

		hlines = [[] for _ in range(rows)]
		for i in range(rows):
			hlines[i] = [[] for _ in range(cols)]

		# s_h = [d.game.eq_utility_seller(), d.game.utility_without_pd_seller()]
		# b_h = [d.game.eq_utility_buyer(), d.game.utility_without_pd_buyer()]
		s_h = [d.game.utility_without_pd_seller()]
		b_h = [d.game.utility_without_pd_buyer()]

		for j in range(cols):
			hlines[1][j] = s_h		
			hlines[0][j] = b_h
	plot_matrix_subplots([b,s], hlines)


def plot_matrix_subplots(a, hlines = None, vlines = None, row_labels=None, col_labels=None):
	rows = len(a)
	cols = len(a[0])
	total_plots = rows * cols

	# Calculate the optimal number of rows and columns
	max_dim = math.ceil(math.sqrt(total_plots))
	rows = min(max_dim, rows)
	cols = min(max_dim, math.ceil(total_plots / rows))

	# Calculate the figure size based on the number of rows and columns
	fig_width = 4 * cols
	fig_height = 4 * rows

	fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

	for i in range(rows):
		for j in range(cols):
			ax = axes[i][j]
			if i < len(a) and j < len(a[i]):
				ax.plot(range(len(a[i][j])), a[i][j])
			if hlines is not None:
				for hline in hlines[i][j]:
					ax.axhline(hline, color='red', linestyle='--')
			if i == 0 and (col_labels is not None):
				ax.set_title(col_labels[j])
			if j == 0 and (row_labels is not None):
				ax.set_ylabel(row_labels[i])

	        # if(i == 0 or j == 0):
		       #  # Set row and column labels if provided
		       #  if row_labels and col_labels:
		       #      ax.set_title(f'({row_labels[i]}, {col_labels[j]})')
		       #  elif row_labels:
		       #      ax.set_title(f'({row_labels[i]}, {j})')
		       #  elif col_labels:
		       #      ax.set_title(f'({i}, {col_labels[j]})')
		       #  else:
		       #      ax.set_title(f'({i}, {j})')

	plt.tight_layout()
	plt.show()



def main():
	pr_high = 0.5
	v_low = 5
	v_high = 15
	cost_evade = 5

	dynamics_names = ['randomStrategy','checkForSignalsUsage','Exp3WithoutSignals','alwaysBuyWhenAffordable']
	extra_args = [[[pr_high,1-pr_high]],[1.0],[],[]]

	price_disc_dynamics = create_price_disc_instance(pr_high,v_high,v_low,cost_evade, dynamics_names,extra_args)

	num_runs = 4
	rewards = []
	T = 2000
	for i in range(num_runs):
		rewards.append(price_disc_dynamics.run(T))
	plot_cum_buyer_seller_utilities(rewards, price_disc_dynamics)
	# print(rewards)



main()

	

