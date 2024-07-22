from dynamics import *
import statistics


# Returns the cumulative average of a list
def cumulative_average(l):
    cumulative_sum = 0
    cumulative_count = 0
    cumulative_avg = []

    for num in l:
        cumulative_sum += num
        cumulative_count += 1
        cumulative_avg.append(cumulative_sum / cumulative_count)

    return cumulative_avg


# Initialize an instance of the price discrimination game dynamics game
# pr_high,v_high,c_low,cost_evade are parameters for the price discrimination game
# dynamics_name: list of names of classes for dynamics in each interaction of the game
# extra_args: extra_args[i] is the list of extra arguments to be passed to initialize an 
#	object of class dynamics_names[i]
def create_price_disc_instance(pr_high, v_high, v_low, cost_evade, dynamics_names, extra_args, T=10000,
                               contexts=None):
    price_disc_game = PriceDiscriminationGame(pr_high, v_high, v_low, cost_evade)
    if contexts is not None:
        price_disc_game = GameWithContextualActions(price_disc_game, contexts)
    ind_nature = 0
    ind_buyer = 0
    ind_seller = 0
    num_interactions = 0
    if contexts is not None:
        ind_buyer = price_disc_game.base_game.reward_ind_buyer()
        ind_seller = price_disc_game.base_game.reward_ind_seller()
        num_interactions = price_disc_game.base_game.num_interactions()
    else:
        ind_buyer = price_disc_game.reward_ind_buyer()
        ind_seller = price_disc_game.reward_ind_seller()
        num_interactions = price_disc_game.num_interactions()
    inds = [ind_nature, ind_buyer, ind_seller, ind_buyer]

    dynamics = []
    for i in range(num_interactions):
        class_name = globals()[dynamics_names[i]]
        ind = inds[i]
        arg = [price_disc_game, i, ind] + extra_args[i]
        dynamics.append(class_name(*arg))

    return GameDynamics(price_disc_game, dynamics)


# Takes rewards in each round of game run for each player
# and optionally the dynamics object
# Plots the cumulative average rewards over time
# optionally indicates the equilibrium or without price discrimination utilities
def plot_cum_buyer_seller_utilities(r, d=None):
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

        game = d.game
        if game.is_contextual:
            game = game.base_game
        s_h = [game.eq_utility_seller(), game.utility_without_pd_seller()]
        b_h = [game.eq_utility_buyer(), game.utility_without_pd_buyer()]

        # s_h = [d.game.utility_without_pd_seller()]
        # b_h = [d.game.utility_without_pd_buyer()]

        for j in range(cols):
            hlines[1][j] = s_h
            hlines[0][j] = b_h
    print(len(b), len(b[0]))
    plot_matrix_subplots([b, s], hlines, None, ['Buyer', 'Seller'])


# Creates a matrix of subplots. The (i,j)th subplot plots
# a[i][j] against range(len(a[i][j]))
def plot_matrix_subplots(a, hlines=None, vlines=None, row_labels=None, col_labels=None, subtitles=None):
    rows = len(a)
    cols = len(a[0])
    total_plots = rows * cols
    print(rows, cols)

    # Calculate the optimal number of rows and columns
    # max_dim = math.ceil(math.sqrt(total_plots))
    # rows = min(max_dim, rows)
    # cols = min(max_dim, math.ceil(total_plots / rows))

    # Calculate the figure size based on the number of rows and columns
    fig_width = 3 * cols
    fig_height = 3 * rows

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.plot(range(len(a[i][j])), a[i][j])
            if hlines is not None:
                for hline in hlines[i][j]:
                    ax.axhline(hline, color='red', linestyle='--')
            if i == 0 and (col_labels is not None):
                ax.set_title(col_labels[j])
            if j == 0 and (row_labels is not None):
                ax.set_ylabel(row_labels[i])
            if (subtitles is not None):
                ax.set_title(subtitles[i][j])

    plt.tight_layout()
    plt.show()


# Takes actions in each round of game run for each player
# Plots the average frequency of each action profile over time
def plot_actions_frequency(actions):
    T = len(actions)

    def tuple_to_index(i, j, k):
        return (i + 2 * j + 4 * k)

    counts = np.zeros((8, T))

    for t in range(T):
        a = actions[t]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if a[0] == i and a[1] == j and a[2] == k:
                        counts[tuple_to_index(i, j, k)][t] += 1

    subplots = [[], []]
    hlines = [[], []]
    titles = [[], []]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = cumulative_average(counts[tuple_to_index(i, j, k)].tolist())
                subplots[i].append(cumulative_average(counts[tuple_to_index(i, j, k)].tolist()))
                hlines[i].append([statistics.mean(p[-50:])])
                s_i = 'high val'
                s_j = 'high sig'
                s_k = 'low price'
                if i == 1:
                    s_i = 'low val'
                if j == 1:
                    s_j = 'low sig'
                if k == 1:
                    s_k = 'high price'
                t = s_i + ',' + s_j + ',' + s_k
                titles[i].append(t)
                print(t, statistics.mean(p[-50:]))

    plot_matrix_subplots(subplots, hlines, None, None, None, titles)


def plot_contextual_actions_frequency(actions):
    T = len(actions[0])
    counts = np.zeros((4, T))

    for t in range(T):
        a = actions[0][t]
        counts[a[2]][t] += 1

    subplots = [[]]
    hlines = [[]]
    titles = [['always low price', 'high sig: low price, low_sig: high price',
               'high sig: high price, low_sig: low price', 'always high price']]
    for i in range(4):
        p = cumulative_average(counts[i].tolist())
        subplots[0].append(p)
        hlines[0].append([statistics.mean(p[-50:])])

    plot_matrix_subplots(subplots, hlines, None, None, None, titles)


def main():
    pr_high = 0.5
    v_low = 5
    v_high = 15
    cost_evade = 5

    # For without signal usage, set is_contextual = False
    is_contextual = True

    # dynamics_names = ['randomStrategy', 'checkForSignalsUsage', 'EXP3_IX', 'alwaysBuyWhenAffordable']
    dynamics_names = ['randomStrategy', 'estimateProbPDSeller', 'EXP3_IX', 'alwaysBuyWhenAffordable']
    extra_args = [[[pr_high, 1 - pr_high]], [1.0, is_contextual], [], [is_contextual]]
    contexts = [[], [], [1], [2]]
    if not is_contextual:
        contexts = None

    T = 2000000
    # T = 200000
    price_disc_dynamics = create_price_disc_instance(pr_high, v_high, v_low, cost_evade, dynamics_names, extra_args, T,
                                                     contexts)
    num_runs = 1
    rewards = []

    actions = []
    for i in range(num_runs):
        a, r = price_disc_dynamics.run(T)
        rewards.append(r)
        actions.append(a)

    # To plot running average of rewards for each run
    plot_cum_buyer_seller_utilities(rewards, price_disc_dynamics)

    # To plot frequencies of action profiles for each run
    plot_contextual_actions_frequency(actions[-10000:])

    # To plot regret of the first run
    seller_dynamic = price_disc_dynamics.dynamics[2]
    avg_reg = seller_dynamic.avg_regret(actions[0])
    plt.plot(range(avg_reg.shape[0]), avg_reg)
    plt.show()

    # Plotting regret of never price discriminating
    # This corresponds to action 3
    actions_non_pd = actions[0]
    for i in range(len(a[0])):
        actions_non_pd[i][2] = 3
    avg_reg_non_pd = seller_dynamic.avg_regret(actions_non_pd)
    plt.plot(range(avg_reg_non_pd.shape[0]), avg_reg_non_pd)
    plt.show()


main()
