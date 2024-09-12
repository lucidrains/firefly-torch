import torch
import einx # s - species, p - population, i - population source, j - population target, t - tournament participants, d - dimension

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# hyperparameters

steps = 5000
species = 4
population_size = 1000
dimensions = 10
lower_bound = -4.
upper_bound = 4.
mix_species_every = 25

beta0 = 2.           # exploitation factor, moving fireflies of low light intensity to high
gamma = 1.           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
alpha = 0.1          # exploration factor
alpha_decay = 0.95   # exploration decay each step

# genetic algorithm related

use_genetic_algorithm = True
breed_every = 10
tournament_size = 50
num_children = 100

assert tournament_size <= population_size
assert num_children <= population_size

# settings

use_cuda = True
verbose = True

cost_function = rosenbrock

# main algorithm

fireflies = torch.zeros((species, population_size, dimensions)).uniform_(lower_bound, upper_bound)

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

    # have species intermix every so often

    if species > 1 and (step % mix_species_every) == 0:
        midpoint = population_size // 2
        fireflies, fireflies_rotate = fireflies[:, :midpoint], fireflies[:, midpoint:]
        fireflies = torch.cat((fireflies, torch.roll(fireflies_rotate, shifts = 1, dims = (0,))), dim = 1)

    # maybe genetic algorithm

    if not use_genetic_algorithm or (step % breed_every) != 0:
        continue

    # use the most effective genetic algorithm - tournament style

    cost = cost_function(fireflies)
    fitness = 1. / cost

    batch_randperm = torch.randn((species, num_children, population_size), device = device).argsort(dim = -1)
    tournament_indices = batch_randperm[..., :tournament_size]

    tournament_participants = einx.get_at('s [p], s c t -> s c t', fitness, tournament_indices)
    winners_per_tournament = tournament_participants.topk(2, dim = -1).indices

    # breed the winners - w = 2

    parent1, parent2 = einx.get_at('s [p] d, s c parents -> parents s c d', fireflies, winners_per_tournament)

    # do a uniform crossover

    crossover_mask = torch.rand_like(parent1) < 0.5

    children = torch.where(crossover_mask, parent1, parent2)

    # sort the fireflies by fitness and replace the worst performing with the new children

    _, sorted_indices = cost.sort(dim = -1)
    sorted_fireflies = einx.get_at('s [p] d, s sorted -> s sorted d', fireflies, sorted_indices)

    fireflies = torch.cat((sorted_fireflies[:, :-num_children],  children), dim = -2)

# print solution

fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

costs = cost_function(fireflies)
sorted_costs, sorted_indices = costs.sort(dim = -1)

fireflies = fireflies[sorted_indices]

print(f'best performing firefly for rosenbrock: {sorted_costs[0]:.3f}: {fireflies[0]}')
