from dynamics_forc import *
import statistics
import seaborn as sns
import pdb

# Initialize an instance of the price discrimination game dynamics game
# pr_high,v_high,c_low,cost_evade are parameters for the price discrimination game
# dynamics_name: list of names of classes for dynamics in each interaction of the game
# player_args: player_args[i] is the list of extra arguments to be passed to initialize an 
#	object of class dynamics_names[i]

def create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, dynamics_names, player_args, T, contexts):
    price_disc_game = PriceDiscriminationGame(pr_high, v_high, v_low, cost_evade, n)

    ind_nature = 0
    ind_buyer = 0
    ind_seller = 0
    num_interactions = 0
    
    ind_buyer = price_disc_game.reward_ind_buyer()
    ind_seller = price_disc_game.reward_ind_seller()
    num_interactions = price_disc_game.num_interactions()
    inds = [ind_nature, ind_buyer, ind_seller, ind_buyer]

    dynamics = []
    for i in range(num_interactions):
        class_name = globals()[dynamics_names[i]]
        ind = inds[i]
        arg = [price_disc_game, i, ind] + player_args[i] + [contexts[i]]
        dynamics.append(class_name(*arg))

    return GameDynamics(price_disc_game, dynamics)

'''
Plotting Functions
'''

# Code for Figure 1 (ordering of buyer/seller utilitis in one-shot game)
def plot_order_of_utilities():
    fig,axs = plt.subplots(1,2,figsize=(10,3))
    
    # pd arguments
    dynamics_names = [nature_strat, signal_strat, 'CExp3', buy_strat]
    player_args = [[[pr_high, 1 - pr_high]], [], [alpha], []]
    contexts = [[], [], [], []]
    alphas = np.linspace(0,1,10000)
    large_theta_low_v_high = 10
    large_theta_low_cost_evade = 2.5
    large_theta_low_game = create_price_disc_instance(pr_high, large_theta_low_v_high, v_low, large_theta_low_cost_evade, n, dynamics_names, player_args, T, contexts).game
    
    large_theta_low_us = [large_theta_low_game.eq_utility_seller(alpha) for alpha in alphas]
    large_theta_low_ub = [large_theta_low_game.eq_utility_buyer(alpha) for alpha in alphas]
    
    small_theta_low_v_high = 15
    small_theta_low_cost_evade = 5
    small_theta_low_game = create_price_disc_instance(pr_high, small_theta_low_v_high, v_low, small_theta_low_cost_evade, n, dynamics_names, player_args, T, contexts).game

    small_theta_low_us = [small_theta_low_game.eq_utility_seller(alpha) for alpha in alphas]
    small_theta_low_ub = [small_theta_low_game.eq_utility_buyer(alpha) for alpha in alphas]


    sns.lineplot(x=alphas[0:int(len(alphas)/2)],y=large_theta_low_ub[0:int(len(alphas)/2)],label='buyer',ax=axs[0],color=palette[0])
    sns.lineplot(x=alphas[int(len(alphas)/2):],y=large_theta_low_ub[int(len(alphas)/2):], ax=axs[0],color=palette[0])
    sns.lineplot(x=alphas[0:int(len(alphas)/2)],y=large_theta_low_us[0:int(len(alphas)/2)],label='seller',ax=axs[0],color=palette[1])
    sns.lineplot(x=alphas[int(len(alphas)/2):],y=large_theta_low_us[int(len(alphas)/2):],ax=axs[0],color=palette[1])
    large_theta_low_state = '$\\theta_l \\geq \\mu\\theta_h$'
    
    sns.lineplot(x=alphas[0:int(len(alphas)/2)],y=small_theta_low_ub[0:int(len(alphas)/2)],label='buyer',ax=axs[1],color=palette[0])
    sns.lineplot(x=alphas[int(len(alphas)/2):],y=small_theta_low_ub[int(len(alphas)/2):],ax=axs[1],color=palette[0])
    sns.lineplot(x=alphas[0:int(len(alphas)/2)],y=small_theta_low_us[0:int(len(alphas)/2)],label='seller',ax=axs[1],color=palette[1])
    sns.lineplot(x=alphas[int(len(alphas)/2):],y=small_theta_low_us[int(len(alphas)/2):],ax=axs[1],color=palette[1])
    small_theta_low_state = '$\\theta_l < \\mu\\theta_h$'

    for ax in axs:
        ax.set(xticks=np.linspace(0,1,3), xticklabels=[0, '$\\alpha^*$', 1])
        ax.set_xlabel('$\\alpha$', fontsize=12)
        ax.set_ylabel('utility', fontsize=12)
        ax.legend()

    sns.despine()
    fig.tight_layout()
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
    fig, axs = plt.subplots(1, len(dynamic_types), figsize=(15,3))
    labels = ['always high price', 'always low price', 'PD', 'revPD']
    for i, dynamic_type in enumerate(dynamic_types):
        ax=axs[i]
        frequencies, action_types = generate_seller_action_frequencies(actions[i], dynamic_type, i)
        for j in range(len(frequencies)):
            sns.lineplot(x=np.arange(T), y=frequencies[j], color=palette[j], label=labels[j], ax=ax)
        ax.set_xlabel(f't')
        ax.set_ylabel('action frequency')
        ax.set_title(f'{dynamic_types[i]}')
        ax.legend()
    
    sns.despine()
    fig.tight_layout()
    plt.show()

# Code for Figure 2 -- plots buyer/seller utilities for a seller playing various algorithms against a CBER buyer 
def plot_utilities(rewards, dynamics):
    print('plotting utilities')
    num_players = 2
    fig, axs = plt.subplots(num_players, len(dynamic_types), figsize=(15, 5))
    for i in range(len(dynamic_types)):
        game = dynamics[i].game
        seller_utilities = []
        buyer_utilities = []
        seller_utilities.append(cumulative_average(rewards[i][2]))
        buyer_utilities.append(cumulative_average(rewards[i][1]))

        sns.lineplot(x=np.arange(T),y=seller_utilities[0], ax=axs[0][i], color=palette[0], label='Repeated PD')
        sns.lineplot(x=np.arange(T),y=buyer_utilities[0], ax=axs[1][i], color=palette[0], label='Repeated PD')
        
        axs[0][i].hlines(y=game.eq_utility_seller(1),xmin=0,xmax=T,linestyles='dashed',label='($\\alpha$=1)-PD',color=palette[1])
        axs[0][i].hlines(y=game.eq_utility_seller(0),xmin=0,xmax=T,linestyles='dashed',label='($\\alpha$=0)-PD',color=palette[2])
        axs[0][i].hlines(y=game.eq_utility_seller(cost_evade/(v_high-v_low)),xmin=0,xmax=T,linestyles='dashed',label='($\\alpha$=$\\alpha^*$)-PD',color=palette[4])
        axs[0][i].set_xlabel('t')
        axs[0][i].set_ylabel('utility')
        axs[0][i].set_title(f'{dynamic_types[i]} seller')
        axs[0][i].legend()

        axs[1][i].hlines(y=game.eq_utility_buyer(1),xmin=0,xmax=T,linestyles='dashed',label='($\\alpha$=1)-PD',color=palette[1])
        axs[1][i].hlines(y=game.eq_utility_buyer(0),xmin=0,xmax=T,linestyles='dashed',label='($\\alpha$=0)-PD',color=palette[2])
        axs[0][i].set_xlabel('t')
        axs[1][i].set_ylabel('utility')
        axs[1][i].set_title('CBER buyer')
        axs[1][i].legend()
        
    sns.despine()
    fig.tight_layout()
    plt.show()

# Code for Figure 3 -- plots convergence of CBER's estimator over rounds
def plot_alpha_hats(alpha_hats):
    print('plotting alpha hats')
    fig, ax = plt.subplots(1, 1, figsize=(4,2))
    for i in range(len(dynamic_types)):
        sns.lineplot(x=np.arange(T), y=alpha_hats[i], color=palette[i], ax=ax, label=dynamic_types[i])
    ax.legend(title='Seller Algorithm', title_fontsize=8, fontsize=8)
    ax.set_xlabel(f't', fontsize=10)
    ax.set_ylabel('$\\hat{\\alpha_t}$', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    fig.suptitle('CBER buyer', fontsize=8)
    sns.despine()
    fig.tight_layout()
    plt.show()

def plot_regret(actions, seller_dynamics):
    avg_regret = seller_dynamics.avg_regret(actions)
    fig, ax = plt.subplots()
    sns.lineplot(x=range(avg_regret.shape[0]), y=avg_regret, color=palette[0])
    ax.set_xlabel('t')
    ax.set_ylabel('Average Regret')
    sns.despine()
    fig.tight_layout()
    plt.show()

# Runs Repeated PD protocol
def create_dynamics(price_strat, is_contextual):
    dynamics_names = [nature_strat, signal_strat, price_strat, buy_strat]
    player_args = [[[pr_high, 1 - pr_high]], [], [alpha], []]
    contexts = [[], [], [1], [2]] if is_contextual else [[], [], [], []]
    repeatedPD = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, n, dynamics_names, player_args, T, contexts)
    r, a, a_hats = repeatedPD.run(T)
    return r, a, a_hats, repeatedPD

# Generates all plots
def generate_plots():
    # Figure 1
    plot_order_of_utilities()
    
    # Other figures
    rewards = []
    actions = []
    alpha_hats = []
    player_dynamics = []
    for i, dynamic_name in enumerate(dynamic_types):
        r, a, a_hats, repeatedPD = create_dynamics(price_strat=price_strats[i], is_contextual=contextual[i])
        rewards.append(r)
        actions.append(a)
        alpha_hats.append(a_hats)
        player_dynamics.append(repeatedPD)
    
    # Figure 2
    plot_utilities(rewards, player_dynamics)

    # Figure 3
    plot_alpha_hats(alpha_hats)

    # Figure 4
    #plot_seller_action_frequencies(actions)

    plot_regret(actions[0], player_dynamics[0].dynamics[2])

if __name__=='__main__':
    
    # Parameters 
    pr_high = 0.5
    v_low = 5
    v_high = 15
    cost_evade = 5
    n = 5
    perturb_prob = 0
    alpha = cost_evade / (v_high - v_low)
    num_estimators = 1
    palette = sns.color_palette('muted')
    T = 50000
    num_runs = 1

    # Flags 
    perturb = False
    
    # Player Strategies
    nature_strat = 'natureStrategy'
    signal_strat = 'consistentPD'
    price_strats = ['CExp3', 'CExp3', 'alphaPDStrategy']
    buy_strat = 'buyStrategy'
    
    dynamic_types = ['Exp3', 'CExp3', '$\\alpha^*$-PD']
    contextual = [False, True, False]

    generate_plots()
    
