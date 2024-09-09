import torch
import einx # s - species, p - population, i - population source, j - population target, d - dimension

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# constants

steps = 1000
species = 4
population_size = 100
dimensions = 4
lower_bound = -4.
upper_bound = 4.
mix_species_every = 25

beta0 = 2.           # exploitation factor, moving fireflies of low light intensity to high
gamma = 1.           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
alpha = 0.1          # exploration factor
alpha_decay = 0.95   # exploration decay each step

use_cuda = True
verbose = True

cost_function = rosenbrock

# main algorithm

fireflies = torch.zeros((species, population_size, dimensions)).uniform_(lower_bound, upper_bound)

# maybe use cuda

if torch.cuda.is_available() and use_cuda:
    fireflies = fireflies.cuda()

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

# print solution

fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

costs = cost_function(fireflies)
sorted_costs, sorted_indices = costs.sort(dim = -1)

fireflies = fireflies[sorted_indices]

print(f'best performing firefly for rosenbrock: {sorted_costs[0]:.3f}: {fireflies[0]}')
