import fire
import torch
import einx # s - colonies, p - population, i - population source, j - population target, t - tournament participants, d - dimension

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# hyperparameters

@torch.inference_mode()
def main(
    steps = 5000,
    colonies = 4,
    population_size = 1000,
    dimensions = 12,      # set this to something lower (2-10) for fireflies without sexual reproduction to solve
    lower_bound = -4.,
    upper_bound = 4.,
    migrate_every = 100,

    beta0 = 2.,           # exploitation factor, moving fireflies of low light intensity to high
    gamma = 1.,           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
    alpha = 0.1,          # exploration factor
    alpha_decay = 0.995,  # exploration decay each step

    # genetic algorithm related

    use_genetic_algorithm = False,  # turn on genetic algorithm, for comparing with non-sexual fireflies
    breed_every = 10,
    tournament_size = 100,
    num_children = 500
):

    assert tournament_size <= population_size
    assert num_children <= population_size

    # settings

    use_cuda = True
    verbose = True

    cost_function = rosenbrock

    # main algorithm

    fireflies = torch.zeros((colonies, population_size, dimensions)).uniform_(lower_bound, upper_bound)

    # maybe use cuda

    if torch.cuda.is_available() and use_cuda:
        fireflies = fireflies.cuda()

    device = fireflies.device

    # iterate

    for step in range(steps):

        # cost, which is inverse of light intensity

        costs = cost_function(fireflies)

        if verbose:
            print(f'{step}: {costs.amin():.5f}')

        # fireflies with lower light intensity (high cost) moves towards the higher intensity (lower cost)

        move_mask = einx.greater('s i, s j -> s i j', costs, costs)

        # get vectors of fireflies to one another
        # calculate distance and the beta

        delta_positions = einx.subtract('s j d, s i d -> s i j d', fireflies, fireflies)

        distance = delta_positions.norm(dim = -1)

        betas = beta0 * (-gamma * distance ** 2).exp()

        # calculate movements

        attraction = einx.multiply('s i j, s i j d -> s i j d', move_mask * betas, delta_positions)
        random_walk = alpha * (torch.rand_like(fireflies) - 0.5) * (upper_bound - lower_bound)

        # move the fireflies

        fireflies += einx.sum('s i j d -> s i d', attraction) + random_walk

        fireflies.clamp_(min = lower_bound, max = upper_bound)

        # decay exploration factor

        alpha *= alpha_decay

        # have colonies migrate every so often

        if colonies > 1 and migrate_every > 0 and (step % migrate_every) == 0:
            midpoint = population_size // 2
            fireflies, fireflies_rotate = fireflies[:, :midpoint], fireflies[:, midpoint:]
            migrate_indices = torch.randperm(colonies, device = device)
            fireflies = torch.cat((fireflies, fireflies_rotate[migrate_indices]), dim = 1)

        # maybe genetic algorithm

        if not use_genetic_algorithm or (step % breed_every) != 0:
            continue

        # use the most effective genetic algorithm - tournament style

        cost = cost_function(fireflies)
        fitness = 1. / cost

        batch_randperm = torch.randn((colonies, num_children, population_size), device = device).argsort(dim = -1)
        tournament_indices = batch_randperm[..., :tournament_size]

        participant_fitnesses = einx.get_at('s [p], s c t -> s c t', fitness, tournament_indices)
        winner_tournament_ids = participant_fitnesses.topk(2, dim = -1).indices

        winning_firefly_indices = einx.get_at('s c [t], s c parents -> s c parents', tournament_indices, winner_tournament_ids)

        # breed the top two winners of each tournament

        parent1, parent2 = einx.get_at('s [p] d, s c parents -> parents s c d', fireflies, winning_firefly_indices)

        # do a uniform crossover

        crossover_mask = torch.rand_like(parent1) < 0.5

        children = torch.where(crossover_mask, parent1, parent2)

        # sort the fireflies by cost and replace the worst performing with the new children

        _, sorted_indices = cost.sort(dim = -1)
        sorted_fireflies = einx.get_at('s [p] d, s sorted -> s sorted d', fireflies, sorted_indices)

        fireflies = torch.cat((sorted_fireflies[:, :-num_children],  children), dim = -2)

    # print solution

    fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

    costs = cost_function(fireflies)
    sorted_costs, sorted_indices = costs.sort(dim = -1)

    fireflies = fireflies[sorted_indices]

    print(f'best performing firefly for rosenbrock with {dimensions} dimensions: {sorted_costs[0]:.3f}: {fireflies[0]}')

# main

if __name__ == '__main__':
    fire.Fire(main)
