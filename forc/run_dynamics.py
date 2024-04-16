from dynamics import *
import statistics
import seaborn as sns
import pdb
import os
import os.path

# Initialize an instance of the price discrimination game dynamics game
# pr_high,v_high,c_low,cost_evade are parameters for the price discrimination game
# dynamics_name: list of names of classes for dynamics in each interaction of the game
# player_args: player_args[i] is the list of extra arguments to be passed to initialize an 
#	object of class dynamics_names[i]
def create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T,
                               strategies=None, num_dynamics=None, player_args=None, contexts=None):
    price_disc_game = PriceDiscriminationGame(pr_high, v_high, v_low, cost_evade, n, T)

    ind_nature = 0
    ind_buyer = 0
    ind_seller = 0
    num_interactions = 0
    
    ind_buyer = price_disc_game.reward_ind_buyer()
    ind_seller = price_disc_game.reward_ind_seller()
    num_interactions = price_disc_game.num_interactions()
    inds = [ind_nature, ind_buyer, ind_seller, ind_buyer]

    dynamics = []
    if strategies:
        for interaction_ind in range(num_interactions):
            class_name = globals()[dynamics_names[interaction_ind]]
            player_ind = inds[interaction_ind]
            args = ([price_disc_game, interaction_ind, player_ind] + 
                    [strategies[interaction_ind]] + 
                    [num_dynamics[interaction_ind]] + 
                    [player_args[interaction_ind]] +
                    [contexts[interaction_ind]])
            dynamics.append(class_name(*args))

    return GameDynamics(price_disc_game, dynamics)

'''
Plotting Functions
'''

# Code for Figure 1 (ordering of buyer/seller utilities in one-shot game)
def plot_order_of_utilities():
    fig,axs = plt.subplots(1, 2, figsize=(10, 3))
    
    alphas = np.linspace(0, 1, 10000)
    
    states = ['$\\theta_l \\geq \\mu\\theta_h$', '$\\theta_l < \\mu\\theta_h$']
    params = [{'v_high': 10, 'cost_evade': 2.5}, {'v_high': 15, 'cost_evade': 5}]
    
    for i,state in enumerate(states):
        v_high = params[i]['v_high']
        cost_evade = params[i]['cost_evade']
        game = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T).game
        seller_utilities = [game.eq_utility('Seller', alpha) for alpha in alphas]
        buyer_utilities = [game.eq_utility('High Value Buyer', alpha) for alpha in alphas]
        
        alpha_discontinuity = np.where(np.diff(seller_utilities) < 0)[0].item()
        sns.lineplot(x=alphas,y=buyer_utilities, label='high value buyer',
                     ax=axs[i], color=muted_palette[0])
        sns.lineplot(x=alphas[0:alpha_discontinuity+1],
                     y=seller_utilities[0:alpha_discontinuity+1],
                     label='seller', ax=axs[i], color=muted_palette[1])
        sns.lineplot(x=alphas[alpha_discontinuity+1:],
                     y=seller_utilities[alpha_discontinuity+1:], ax=axs[i],
                     color=muted_palette[1])

    for ax in axs:
        ax.set(xticks=np.linspace(0, 1, 3), xticklabels=[0, '$\\alpha^*$', 1])
        ax.tick_params(axis='both', labelsize=axis_fs)
        ax.set_xlabel('$\\alpha$', fontsize=label_fs)
        ax.set_ylabel('Utility', fontsize=label_fs)
        ax.legend(fontsize=legend_fs)

    sns.despine()
    fig.tight_layout()
    plt.savefig(path+'order_of_utilities.pdf')
    plt.show()

# Computes frequencies of seller's actions over time horizon
def generate_seller_action_frequencies(actions, dynamic_type, j):
    action_types = ['high price', 'low price', 'PD', 'revPD']
    counts = np.zeros((len(action_types), T))
            
    for t in range(T):
        if t % 1000==0:
            print('t',t)
        if all(x==1 for x in actions[t][2]):
            counts[0][t] = counts[0][t-1] + 1 if t > 0 else 1 
            counts[0][t+1:] = [counts[0][t] for _ in range(T-(t+1))]
        elif all(x==0 for x in actions[t][2]):
            counts[1][t] = counts[1][t-1] + 1 if t > 0 else 1 
            counts[1][t+1:] = [counts[1][t] for _ in range(T-(t+1))]
        elif all(x==1 for x in [price + signal for (price, signal) in zip(actions[t][2], actions[t][1])]):
            counts[2][t] = counts[2][t-1] + 1 if t > 0 else 1 
            counts[2][t+1:] = [counts[2][t] for _ in range(T-(t+1))]
        elif all(x % 2 == 0 for x in [price + signal for (price, signal) in zip(actions[t][2], actions[t][1])]):
            counts[3][t] = counts[3][t-1] + 1 if t > 0 else 1 
            counts[3][t+1:] = [counts[3][t] for _ in range(T-(t+1))]
    
    frequencies = np.divide(counts, np.arange(1,T+1))

    return frequencies, action_types

# Code for Figure 4 -- plots frequencies of seller's actions over time horizon
def plot_seller_action_frequencies(actions):
    print('plotting seller action frequencies')
    fig, axs = plt.subplots(1, len(price_strat_labels), figsize=(15, 5))
    labels = ['high price', 'low price', 'PD', 'reversePD']
    for i, dynamic_type in enumerate(price_strat_labels):
        print(dynamic_type)
        ax=axs[i]
        frequencies, action_types = generate_seller_action_frequencies(actions[i], dynamic_type, i)
        for j in range(len(frequencies)):
            sns.lineplot(x=np.arange(T), y=frequencies[j], color=muted_palette[j], label=labels[j], ax=ax)
        ax.set_xlabel('t', fontsize=label_fs)
        ax.set_ylabel('Action Frequency', fontsize=label_fs)
        ax.set_title(f'{price_strat_labels[i]}', fontsize=label_fs)
        ax.tick_params(axis='both', labelsize=axis_fs)
        ax.legend(fontsize=legend_fs)
    
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'action_frequencies.pdf')
    plt.show()

# Code for Figure 2 -- plots buyer/seller utilities for a seller playing various algorithms against a CBER buyer 
#def plot_utilities(rewards, dynamics):
#    print('plotting utilities')
#    players = {0:'seller', 1:'buyer'}
#    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
#    fig, axs = plt.subplots(len(players), len(price_strat_labels), figsize=(15, 5))
#    colors = [muted_palette[i] for i in [0, 1, 2, 4]]
#    for i in range(np.shape(rewards)[0]):
#        game = dynamics[i].game
#        utilities = []
#        seller_utilities = cumulative_average(rewards[i][2])
#        buyer_utilities = cumulative_average(rewards[i][1])
#        utilities.append(seller_utilities)
#        utilities.append(buyer_utilities)
#        for j in range(len(players)):
#            sns.lineplot(x=np.arange(T), y=utilities[j], ax=axs[j][i], color=colors[0])
#            for idx, alpha in enumerate(alpha_levels):
#                label = f'($\\alpha$={alpha})-PD' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD'
#                axs[j][i].hlines(y=game.eq_utility(players[j],alpha),xmin=0,xmax=T,linestyles='dashed',label=label,color=colors[idx+1])
#            axs[j][i].set_title(f'{price_strat_labels[i]} {players[j]}') if j==0 else axs[j][i].set_title(f'CBER {players[j]}')
#            axs[1][i].set_xlabel('t', fontsize=label_fs)
#            axs[j][0].set_ylabel('Utility', fontsize=label_fs)
#            axs[j][i].tick_params(axis='both', labelsize=axis_fs)
#            axs[j][i].legend(fontsize=legend_fs)
#    
#    fig.autofmt_xdate()
#    sns.despine()
#    fig.tight_layout()
#    plt.savefig(path+'utilities.pdf')
#    plt.show()

def plot_utilities(rewards, dynamics):
    print('plotting utilities')
    players = {0:'Seller', 1:'Buyer'}
    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
    fig, axs = plt.subplots(nrows=1, ncols=len(players), figsize=(15, 5))
    utility_color_idxs = [0, 2, 3]
    equil_color_idxs = [1, 4, 7]
    game = dynamics[0].game
    for j in range(len(players)):
        player_idx = 1 if players[j] == 'Buyer' else 2
        for i in range(np.shape(rewards)[0]):
            sns.lineplot(x=np.arange(T),
                         y=cumulative_average(rewards[i][player_idx]),
                         ax=axs[j], color=muted_palette[utility_color_idxs[i]],
                         label=price_strat_labels[i])

        for i, alpha in enumerate(alpha_levels):
            label = f'($\\alpha$={alpha})-PD equil.' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD equil.'
            axs[j].hlines(y=game.eq_utility(players[j], alpha), xmin=0, xmax=T, linestyles='dashed', 
                          label=label, color=muted_palette[equil_color_idxs[i]])

        axs[j].set_xlabel('t', fontsize=label_fs)
        axs[j].set_ylabel('Utility', fontsize=label_fs)
        axs[j].tick_params(axis='both', labelsize=axis_fs)
        axs[j].set_title(players[j], fontsize=label_fs)

    axs[0].legend(ncols=2, fontsize=legend_fs)
    axs[1].legend(ncols=2, fontsize=legend_fs)
    #sns.set_context()
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'utilities.pdf')
    plt.show()

def plot_flip_effect():
    print('plotting noise effects')
    flip_probs = np.arange(0, 1, 0.1)
    fig, axs = plt.subplots(nrows=1, ncols=len(price_strat_labels), figsize=(15, 5))
    for i,label in enumerate(price_strat_labels):
        print(label)
        conv_seller_utility = []
        conv_buyer_utility = []
        for pr_flip in flip_probs:
            print('flip prob', pr_flip)
            r = create_dynamics(i, pr_flip=pr_flip, noise_bounds=[0,0])[0]
            conv_seller_utility.append(cumulative_average(r[2])[-1])
            conv_buyer_utility.append(cumulative_average(r[1])[-1])
            
        sns.lineplot(x=flip_probs, y=conv_seller_utility, ax=axs[i],
                     label='seller', color=muted_palette[0])
        sns.lineplot(x=flip_probs, y=conv_buyer_utility, ax=axs[i],
                     label='buyer', color=muted_palette[1])
        
        axs[i].set_xlabel('flip probability')
        axs[i].set_ylabel('Cumulative Average Utility')
        axs[i].set_title(label)
        axs[i].legend()
    
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'flip_effect.pdf')
    plt.show()

def plot_utilities_per_noise_interval(noise_interval):
    players = {0:'seller', 1:'buyer'}
    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
    noise_bounds = np.arange(-1, 1, noise_interval)
    colors = [muted_palette[i] for i in [0, 1, 2, 4]]
    game = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T).game
    fig = plt.figure(layout='constrained', figsize=(15, 5 * len(noise_bounds)))
    subfigs = fig.subfigures(len(noise_bounds), 1)

    for i, noise_bound in enumerate(noise_bounds):
        print('noise bound', noise_bound)
        axs = subfigs[i].subplots(len(players), len(price_strat_labels), sharey=True)
        subfigs[i].suptitle(f'noise interval = {[noise_bounds[i], noise_bounds[i] + noise_interval]}')
        subfigs[i].set_facecolor('0.75')
        rewards = []
        for j, label in enumerate(price_strat_labels):
            print(label)
            r = create_dynamics(j, pr_flip=0, noise_bounds=[noise_bounds[i], noise_bounds[i] + noise_interval])[0]
            rewards.append(r)
        for j, label in enumerate(price_strat_labels):
            utilities = []
            seller_utilities = cumulative_average(rewards[j][2])
            buyer_utilities = cumulative_average(rewards[j][1])
            utilities.append(seller_utilities)
            utilities.append(buyer_utilities)
            for k in range(len(players)):
                sns.lineplot(x=np.arange(T), y=utilities[k], ax=axs[k][j], color=colors[0], label='Repeated PD')
                for idx, alpha in enumerate(alpha_levels):
                    label = f'($\\alpha$={alpha})-PD' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD'
                    axs[k][j].hlines(y=game.eq_utility(players[k],alpha),xmin=0,xmax=T,linestyles='dashed',label=label,color=colors[idx+1])
                if k==0:
                    axs[k][j].set_title(f'{price_strat_labels[j]} {players[k]}', fontsize=8)
                else:
                    axs[k][j].set_title(f'CBER {players[k]}', fontsize=8)
                axs[k][j].set_xlabel('t')
                axs[k][j].set_ylabel('utility')
                axs[k][j].legend()

    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.savefig(path+f'utilities_per_noise_interval.pdf')
    plt.show()

#def plot_cumulative_avg_utility_per_noise_interval(noise_interval):
#    print('plotting noise intervals')
#    players = {0:'seller', 1:'buyer'}
#    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
#    noise_bounds = np.arange(-1, 1, noise_interval)
#    fig, axs = plt.subplots(nrows=1, ncols=len(price_strat_labels), figsize=(15, 3))
#    colors = [muted_palette[i] for i in [0, 1, 2, 4]]
#    for i,label in enumerate(price_strat_labels):
#        print(label)
#        conv_seller_utility = []
#        conv_buyer_utility = []
#        for noise_bound in noise_bounds:
#            print('noise bounds', [noise_bound, noise_bound + noise_interval])
#            r = create_dynamics(i, pr_flip=0, noise_bounds=[noise_bound, noise_bound + noise_interval])[0]
#            conv_seller_utility.append(cumulative_average(r[2])[-1])
#            conv_buyer_utility.append(cumulative_average(r[1])[-1])
#            
#        sns.lineplot(x=noise_bounds, y=conv_seller_utility, ax=axs[i], color=muted_palette[0])
#        #sns.lineplot(x=noise_bounds, y=conv_buyer_utility, ax=axs[i],
#        #             label='buyer', color=muted_palette[1])
#        game = create_dynamics(i, pr_flip=0, noise_bounds=[])[-1].game
#        #for j in range(len(players)):
#        for j in range(1):
#            for idx, alpha in enumerate(alpha_levels):
#                if j==0:
#                    legend_label = f'($\\alpha$={alpha})-PD' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD'
#                else:
#                    legend_label = None
#                axs[i].hlines(y=game.eq_utility(players[j], alpha), xmin=-1, xmax=1,
#                              linestyles='dashed', label=legend_label,
#                              color=colors[idx+1])
#            
#        axs[i].set_xlabel('Estimator Noise Level', fontsize=label_fs)
#        axs[0].set_ylabel('Seller\'s Cumulative \n Average Utility', fontsize=label_fs)
#        axs[i].set_title(label, fontsize=label_fs)
#        axs[i].tick_params(axis='both', labelsize=axis_fs)
#        axs[i].legend()
#    
#    fig.autofmt_xdate()
#    sns.despine()
#    fig.tight_layout()
#    plt.savefig(path + 'cumulative_avg_utility_per_noise_interval.pdf')
#    plt.show()

def plot_cumulative_avg_utility_per_noise_interval(noise_interval):
    print('plotting noise intervals')
    players = {0:'Seller', 1:'Buyer'}
    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
    noise_bounds = np.arange(-1, 1, noise_interval)
    fig, ax = plt.subplots()
    utility_color_idxs = [0, 2, 3]
    equil_color_idxs = [1, 4, 7]
    for i, label in enumerate(price_strat_labels):
        print(label)
        conv_seller_utility = []
        conv_buyer_utility = []
        for noise_bound in noise_bounds:
            print('noise bounds', [noise_bound, noise_bound + noise_interval])
            r = create_dynamics(i, pr_flip=0, noise_bounds=[noise_bound, noise_bound + noise_interval])[0]
            conv_seller_utility.append(cumulative_average(r[2])[-1])
            conv_buyer_utility.append(cumulative_average(r[1])[-1])
        sns.lineplot(x=noise_bounds, y=conv_seller_utility,
                     color=muted_palette[utility_color_idxs[i]], label=label)

    game = create_dynamics(0, pr_flip=0, noise_bounds=[])[-1].game
    for i, alpha in enumerate(alpha_levels):
        legend_label = f'($\\alpha$={alpha})-PD equil.' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD equil.'
        ax.hlines(y=game.eq_utility(players[0], alpha), xmin=-1, xmax=1,
                      linestyles='dashed', label=legend_label,
                      color=muted_palette[equil_color_idxs[i]])
            
    ax.set_xlabel('Estimator Noise Level', fontsize=label_fs)
    ax.set_ylabel('Seller\'s Cumulative \n Average Utility', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    ax.legend(ncols=2, fontsize=legend_fs)
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'cumulative_avg_utility_per_noise_interval.pdf')
    plt.show()

# Code for Figure 3 -- plots convergence of CBER's estimator over rounds
def plot_alphas_vs_alpha_hats(alphas, alpha_hats):
    print('plotting alpha hats')
    alpha_hats = np.squeeze(alpha_hats)
    fig, ax = plt.subplots(1, 1)
    color_idxs = [0, 2, 3]
    for i, label in enumerate(price_strat_labels):
        sns.lineplot(x=np.arange(T), y=alpha_hats[i],
                     color=pastel_palette[color_idxs[i]], label=f'{label} '+'$\\hat{\\alpha}_t$')
        sns.lineplot(x=np.arange(T), y=cumulative_average(alphas[i]), 
                     color=dark_palette[color_idxs[i]], label=f'{label} '+ '$\\overline{\\alpha_t}$', linestyle='--')

    ax.legend(fontsize=legend_fs)
    ax.set_xlabel('t', fontsize=label_fs)
    ax.set_ylabel('$\\hat{\\alpha}_t$ and  $\\overline{\\alpha_t}$', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    fig.suptitle('CBER buyer', fontsize=label_fs)
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'alpha_hats_vs_alphas.pdf')
    plt.show()

def plot_regret(actions, seller_dynamics):
    print('plotting regret')
    avg_regret = seller_dynamics.avg_regret(actions)
    fig, ax = plt.subplots()
    sns.lineplot(x=range(avg_regret.shape[0]), y=avg_regret, color=muted_palette[0])
    ax.set_xlabel('t')
    ax.set_ylabel('Average Regret of Exp3')
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'regret.pdf')
    plt.show()

# Runs Repeated PD protocol
def create_dynamics(price_strat_idx, **kwargs):
    
    price_strat = price_strat_args[price_strat_idx]
    if 'pr_flip' in kwargs:
        pr_flip = kwargs['pr_flip']
    if 'noise_bounds' in kwargs:
        noise_bounds = kwargs['noise_bounds']

    strategies = [nature_strat, signal_strat, price_strat['strat_name'], buy_strat]
    num_dynamics = [1, num_estimators, price_strat['num_dynamics'], 1]
    player_args = [None,
                   {'num_estimators': num_estimators,
                    'pr_estimators': pr_estimators,
                    'pr_flip': pr_flip,
                    'noise_bounds': noise_bounds},
                   price_strat['player_args'],
                   None]

    contexts = [{'inds': None, 'function': None},
                {'inds': None, 'function': 'self._assign_estimators'},
                {'inds': price_strat['contexts']['inds'], 'function': price_strat['contexts']['function']},
                {'inds': None, 'function': None}]
    
    repeated_pd = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T, 
                                             strategies, num_dynamics, player_args, contexts)
    r, a, a_hats, true_as = repeated_pd.run(price_strat_labels[price_strat_idx])
    return r, a, a_hats, true_as, repeated_pd

def create_all_dynamics():
    all_rewards = []
    all_actions = []
    all_alpha_hats = []
    all_alphas = []
    all_dynamics = []
    for i in range(len(price_strat_args)):
        print(price_strat_labels[i])
        rewards, actions, alpha_hats, alphas, repeated_pd = create_dynamics(i, pr_flip=0, noise_bounds=[0,0])
        all_rewards.append(rewards)
        all_actions.append(actions)
        all_alpha_hats.append(alpha_hats)
        all_alphas.append(alphas)
        all_dynamics.append(repeated_pd)
    return all_rewards, all_actions, all_alpha_hats, all_alphas, all_dynamics

# Generates all plots
def generate_plots():

    # self-contained plots
    plot_order_of_utilities()
    plot_cumulative_avg_utility_per_noise_interval(noise_interval=0.1) 
    #plot_flip_effect()
    #plot_utilities_per_noise_interval(noise_interval=0.25)
    
    all_rewards, all_actions, all_alpha_hats, all_alphas, all_dynamics = create_all_dynamics()
    plot_utilities(all_rewards, all_dynamics)
    plot_alphas_vs_alpha_hats(all_alphas, all_alpha_hats)
    plot_seller_action_frequencies(all_actions)
    #plot_regret(all_actions[0], all_dynamics[0].dynamics[2])

if __name__=='__main__':
    # Fix randomness
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    
    # Parameters 
    pr_high = 0.5
    v_low = 5
    v_high = 15
    pr_flip = 0
    cost_evade = 5
    n = 10
    epsilon = 0.01
    alpha = cost_evade / (v_high - v_low) - epsilon
    num_estimators = 1
    pr_estimators = [1, 0]
    T = 20000
    num_runs = 1

    # Player Dynamics
    nature_dynamic = 'natureDynamic'
    signal_dynamic = 'signalDynamic'
    price_dynamic = 'priceDynamic'
    buy_dynamic = 'buyDynamic'
    dynamics_names = [nature_dynamic, signal_dynamic, price_dynamic, buy_dynamic]

    # Player Strategies
    nature_strat = 'randomSelection'
    signal_strat = 'consistentEstimate'
    price_strat = ['Exp3', 'alphaPD']
    buy_strat = 'buyUnderValue'

    # Experiment Settings
    # strategy -- num_dynamics -- player args -- contexts
    price_strat_args = [{'strat_name': 'Exp3',
                         'num_dynamics': 1,
                         'player_args': None, 
                         'contexts': {'inds': [None], 'function': None}},
                         {'strat_name': 'Exp3',
                          'num_dynamics': 2,
                          'player_args': None,
                          'contexts': {'inds': [1], 'function':
                                       'self._assign_dynamics'}},
                         {'strat_name': 'alphaPD',
                          'num_dynamics': 1,
                          'player_args': {'alpha': alpha},
                          'contexts': {'inds': [None], 'function': None}}]
   
    # Plot Settings
    price_strat_labels = ['Exp3', 'CExp3', '$\\alpha^*$-PD']
    path = '/Users/marielwerner/desktop/PrivacyPlots/'
    data_path =  os.getcwd() + '/data/'
    label_fs = 18
    axis_fs = 18
    legend_fs = 12
    muted_palette = sns.color_palette('muted')
    dark_palette = sns.color_palette('dark')
    pastel_palette = sns.color_palette('pastel')
    
    generate_plots()
    
