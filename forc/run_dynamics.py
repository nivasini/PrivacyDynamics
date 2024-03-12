from dynamics import *
import statistics
import seaborn as sns
import pdb

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
    fig,axs = plt.subplots(1, 2, figsize=(10,3))
    
    alphas = np.linspace(0,1,10000)
    
    states = ['$\\theta_l \\geq \\mu\\theta_h$', '$\\theta_l < \\mu\\theta_h$']
    params = [{'v_high': 10, 'cost_evade': 2.5}, {'v_high': 15, 'cost_evade': 5}]
    
    for i,state in enumerate(states):
        v_high = params[i]['v_high']
        cost_evade = params[i]['cost_evade']
        game = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T).game
        seller_utilities = [game.eq_utility('seller', alpha) for alpha in alphas]
        buyer_utilities = [game.eq_utility('buyer', alpha) for alpha in alphas]
        
        alpha_discontinuity = np.where(np.diff(seller_utilities)<0)[0].item()
        sns.lineplot(x=alphas,y=buyer_utilities,label='buyer',ax=axs[i],color=palette[0])
        sns.lineplot(x=alphas[0:alpha_discontinuity+1],
                     y=seller_utilities[0:alpha_discontinuity+1],label='seller',ax=axs[i],color=palette[1])
        sns.lineplot(x=alphas[alpha_discontinuity+1:],
                     y=seller_utilities[alpha_discontinuity+1:],ax=axs[i],color=palette[1])

    for ax in axs:
        ax.set(xticks=np.linspace(0,1,3), xticklabels=[0, '$\\alpha^*$', 1])
        ax.set_xlabel('$\\alpha$', fontsize=12)
        ax.set_ylabel('utility', fontsize=12)
        ax.legend()

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
    fig, axs = plt.subplots(1, len(price_strat_labels), figsize=(15,3))
    labels = ['always high price', 'always low price', 'PD', 'revPD']
    for i, dynamic_type in enumerate(price_strat_labels):
        ax=axs[i]
        frequencies, action_types = generate_seller_action_frequencies(actions[i], dynamic_type, i)
        for j in range(len(frequencies)):
            sns.lineplot(x=np.arange(T), y=frequencies[j], color=palette[j], label=labels[j], ax=ax)
        ax.set_xlabel(f't')
        ax.set_ylabel('action frequency')
        ax.set_title(f'{price_strat_labels[i]}')
        ax.legend()
    
    sns.despine()
    fig.tight_layout()
    plt.savefig(path+'action_frequences.pdf')
    plt.show()

# Code for Figure 2 -- plots buyer/seller utilities for a seller playing various algorithms against a CBER buyer 
def plot_utilities(rewards, dynamics):
    print('plotting utilities')
    players = {0:'seller', 1:'buyer'}
    alpha_levels = [0, 1, cost_evade / (v_high - v_low)]
    fig, axs = plt.subplots(len(players), len(price_strat_labels), figsize=(15, 5))
    colors = [palette[i] for i in [0,1,2,4]]
    for i in range(len(price_strat_labels)):
        game = dynamics[i].game
        utilities = []
        seller_utilities = cumulative_average(rewards[i][2])
        buyer_utilities = cumulative_average(rewards[i][1])
        utilities.append(seller_utilities)
        utilities.append(buyer_utilities)
        for j in range(len(players)):
            sns.lineplot(x=np.arange(T),y=utilities[j],ax=axs[j][i],color=colors[0],label='Repeated PD')
            for idx, alpha in enumerate(alpha_levels):
                label = f'($\\alpha$={alpha})-PD' if alpha in [0, 1] else f'($\\alpha$=$\\alpha^*$)-PD'
                axs[j][i].hlines(y=game.eq_utility(players[j],alpha),xmin=0,xmax=T,linestyles='dashed',label=label,color=colors[idx+1])
            axs[j][i].set_title(f'{price_strat_labels[i]} {players[j]}') if j==0 else axs[j][i].set_title(f'CBER {players[j]}')
            axs[j][i].set_xlabel('t')
            axs[j][i].set_ylabel('utility')
            axs[j][i].legend()
    
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path+'utilities.pdf')
    plt.show()

def plot_noise_effect():
    print('plotting noise effects')
    noise_levels = np.arange(0, 1, 0.1)
    fig, axs = plt.subplots(nrows=1, ncols=len(price_strat_labels), figsize=(15, 5))
    for i in range(len(price_strat_labels)):
        conv_seller_utility = []
        conv_buyer_utility = []
        for noise_level in noise_levels:
            print('noise level', noise_level)
            r, _, _, _ = create_dynamics(i, noise_level=noise_level)
            conv_seller_utility.append(cumulative_average(r[2])[-1])
            conv_buyer_utility.append(cumulative_average(r[1])[-1])
            
        sns.lineplot(x=noise_levels, y=conv_seller_utility, ax=axs[i], label='seller', color=palette[0])
        sns.lineplot(x=noise_levels, y=conv_buyer_utility, ax=axs[i], label='buyer', color=palette[1])
        
        axs[i].set_xlabel('flip probability')
        axs[i].set_ylabel('Achieved Average Utility')
        axs[i].legend()
    
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'noise_levels.pdf')
    plt.show()

# Code for Figure 3 -- plots convergence of CBER's estimator over rounds
def plot_alpha_hats(alpha_hats):
    print('plotting alpha hats')
    fig, ax = plt.subplots(1, 1, figsize=(4,2))
    for i in range(len(price_strat_labels)):
        sns.lineplot(x=np.arange(T), y=alpha_hats[i], color=palette[i], ax=ax, label=price_strat_labels[i])
    ax.legend(title='Seller Algorithm', title_fontsize=8, fontsize=8)
    ax.set_xlabel(f't', fontsize=10)
    ax.set_ylabel('$\\hat{\\alpha_t}$', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    plt.ticklabel_format(useOffset=False)
    fig.suptitle('CBER buyer', fontsize=8)
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path+'alpha_hats.pdf')
    plt.show()

def plot_regret(actions, seller_dynamics):
    print('plotting regret')
    avg_regret = seller_dynamics.avg_regret(actions)
    fig, ax = plt.subplots()
    sns.lineplot(x=range(avg_regret.shape[0]), y=avg_regret, color=palette[0])
    ax.set_xlabel('t')
    ax.set_ylabel('Average Regret')
    fig.autofmt_xdate()
    sns.despine()
    fig.tight_layout()
    plt.savefig(path + 'regret.pdf')
    plt.show()

# Runs Repeated PD protocol
def create_dynamics(price_strat_idx):
    
    price_strat = price_strat_args[price_strat_idx]

    strategies = [nature_strat, signal_strat, price_strat['strat_name'], buy_strat]
    num_dynamics = [1, num_estimators, price_strat['num_dynamics'], 1]
    player_args = [None,
                   {'num_estimators': num_estimators,
                    'pr_estimators': pr_estimators,
                    'pr_flip': pr_flip},
                   price_strat['player_args'],
                   None]

    contexts = [{'inds': None, 'function': None},
                {'inds': None, 'function': 'self._assign_estimators'},
                {'inds': price_strat['contexts']['inds'], 'function': price_strat['contexts']['function']},
                {'inds': None, 'function': None}]
    
    repeated_pd = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, T, 
                                             strategies, num_dynamics, player_args, contexts)
    r, a, a_hats = repeated_pd.run()
    return r, a, a_hats, repeated_pd

# Generates all plots
def generate_plots():

    #plot_order_of_utilities()
    #plot_noise_effect()
    
    rewards = []
    actions = []
    alpha_hats = []
    player_dynamics = []
    for i in range(len(price_strat_args)):
        print(price_strat_labels[i])
        r, a, a_hats, repeated_pd = create_dynamics(i)
        rewards.append(r)
        actions.append(a)
        alpha_hats.append(a_hats)
        player_dynamics.append(repeated_pd)
    
    plot_utilities(rewards, player_dynamics)
    #plot_alpha_hats(alpha_hats)
    plot_regret(actions[0], player_dynamics[0].dynamics[2])
    #plot_seller_action_frequencies(actions)

if __name__=='__main__':
    
    # Parameters 
    pr_high = 0.5
    v_low = 5
    v_high = 15
    pr_flip = 0
    cost_evade = 5
    n = 10
    alpha = cost_evade / (v_high - v_low)
    num_estimators = 1
    pr_estimators = [1, 0]
    palette = sns.color_palette('muted')
    T = 10000
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
    path = '/Users/marielwerner/desktop/'
    
    generate_plots()
    
