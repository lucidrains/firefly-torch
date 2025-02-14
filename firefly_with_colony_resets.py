import fire
import torch
import einx # s - colonies, p - population, i - population source, j - population target, t - tournament participants, d - dimension

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# genetic algorithm functions

def children_from_tournaments(
    fireflies,
    fitness,
    num_children,
    tournament_size
):
    device = fireflies.device

    shape = list(fireflies.shape)
    shape[-2] = num_children

    batch_randperm = torch.randn(shape, device = device).argsort(dim = -1)
    tournament_indices = batch_randperm[..., :tournament_size]

    participant_fitnesses = einx.get_at('... [p], ... c t -> ... c t', fitness, tournament_indices)
    winner_tournament_ids = participant_fitnesses.topk(2, dim = -1).indices

    winning_firefly_indices = einx.get_at('... c [t], ... c parents -> ... c parents', tournament_indices, winner_tournament_ids)

    # breed the top two winners of each tournament

    parent1, parent2 = einx.get_at('... [p] d, ... c parents -> parents ... c d', fireflies, winning_firefly_indices)

    # do a uniform crossover

    crossover_mask = torch.rand_like(parent1) < 0.5

    children = torch.where(crossover_mask, parent1, parent2)
    return children

# hyperparameters

@torch.inference_mode()
def main(
    steps = 5000,
    colonies = 5,
    population_size = 1000,
    dimensions = 15,      # set this to something lower (2-10) for fireflies without sexual reproduction to solve
    lower_bound = -4.,
    upper_bound = 4.,
    migrate_every = 200,
    frac_migrate = 0.1,

    beta0 = 2.,           # exploitation factor, moving fireflies of low light intensity to high
    gamma = 1.,           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
    alpha = 0.1,          # exploration factor
    alpha_decay = 0.995,  # exploration decay each step

    # genetic algorithm related

    use_genetic_algorithm = False,  # turn on genetic algorithm, for comparing with non-sexual fireflies
    breed_every = 5,
    tournament_size = 100,
    num_children = 250,

    # colonies / island resets

    num_colonies_reset = 1,
    reset_colonies_every = 400,
    reset_frac = 0.95,          # need to preserve some of the elite performers
    reset_tournament_size = 25, # when resetting, go for more diversity
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

        if step > 0 and colonies > 1 and migrate_every > 0 and (step % migrate_every) == 0:
            num_migrate = int(population_size * frac_migrate)
            fireflies, fireflies_rotate = fireflies[:, :-num_migrate], fireflies[:, -num_migrate:]
            fireflies_rotate = torch.roll(fireflies_rotate, 1, dims = (0,))
            fireflies = torch.cat((fireflies, fireflies_rotate), dim = 1)

        # maybe genetic algorithm

        if not use_genetic_algorithm or (step % breed_every) != 0:
            continue

        # use the most effective genetic algorithm - tournament style

        cost = cost_function(fireflies)
        fitness = 1. / cost

        children = children_from_tournaments(fireflies, fitness, num_children, tournament_size)

        # sort the fireflies by fitness and replace the worst performing with the new children

        replacement_mask = fitness.argsort(dim = -1).argsort(dim = -1) < num_children

        fireflies[replacement_mask] = einx.rearrange('s p d -> (s p) d', children)

        if step > 0 and 0 < num_colonies_reset < colonies and (step % reset_colonies_every) == 0:
            cost = cost_function(fireflies)
            fitness = 1. / cost

            # determine lowest scoring colonie for reset

            fitness_colonies = fitness.mean(dim = -1)

            sorted_indices = fitness_colonies.sort(dim = -1).indices

            # keep only the best performing colonies

            reset_colonies_indices = sorted_indices[:num_colonies_reset]
            top_colonies_indices = sorted_indices[num_colonies_reset:]

            top_colony_fireflies = fireflies[top_colonies_indices]
            top_colony_fitness = fitness[top_colonies_indices]

            # repopulate new colonies from the aggregated pool of fireflies

            pooled_fireflies = einx.rearrange('s p d -> (s p) d', top_colony_fireflies)
            pooled_fitness = einx.rearrange('s p -> (s p)', top_colony_fitness)

            num_children = int(reset_frac * population_size)
            children = children_from_tournaments(pooled_fireflies, pooled_fitness, num_children * num_colonies_reset, reset_tournament_size)

            reset_colony_fireflies = fireflies[reset_colonies_indices]
            reset_colony_fitness = fitness[reset_colonies_indices]

            replacement_mask = reset_colony_fitness.argsort(dim = -1).argsort(dim = -1) < num_children
            reset_colony_fireflies[replacement_mask] = children

            fireflies = torch.cat((top_colony_fireflies, reset_colony_fireflies), dim = 0)

    # print solution

    fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

    costs = cost_function(fireflies)
    sorted_costs, sorted_indices = costs.sort(dim = -1)

    fireflies = fireflies[sorted_indices]

    print(f'best performing firefly for rosenbrock with {dimensions} dimensions: {sorted_costs[0]:.3f}: {fireflies[0]}')

# main

if __name__ == '__main__':
    fire.Fire(main)
