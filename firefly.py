import torch
import einx

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# constants

steps = 1000
species = 2
population_size = 100
dimensions = 4
lower_bound = -4.
upper_bound = 4.
mix_species_every = 25

beta0 = 2.           # exploitation factor, moving fireflies of low light intensity to high
gamma = 1.           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
alpha = 0.1          # exploration factor
alpha_decay = 0.9    # exploration decay each step

# main algorithm

fireflies = torch.zeros((species, population_size, dimensions)).uniform_(lower_bound, upper_bound)

cost_function = rosenbrock

for step in range(steps):

    # cost, which is inverse of light intensity

    costs = cost_function(fireflies)

    # fireflies with lower light intensity (high cost) moves towards the higher intensity (lower cost)

    move_mask = einx.greater('... i, ... j -> ... i j', costs, costs)

    # get vectors of fireflies to one another
    # calculate distance and the beta

    delta_positions = einx.subtract('... j d, ... i d -> ... i j d', fireflies, fireflies)

    distance = delta_positions.norm(dim = -1)

    # todo - figure out why the difference in fitness does not factor into the beta

    betas = beta0 * (-gamma * distance ** 2).exp()

    # calculate movements

    attraction = einx.multiply('... i j, ... i j d -> ... i j d', move_mask * betas, delta_positions)
    random_walk = alpha * (torch.rand_like(fireflies) - 0.5) * (upper_bound - lower_bound)

    # move the fireflies

    fireflies += einx.sum('... i j d -> ... i d', attraction) + random_walk

    fireflies.clamp_(min = lower_bound, max = upper_bound)

    # have species intermix every so often

    if species > 1 and (step % mix_species_every) == 0:
        midpoint = population_size // 2
        fireflies, fireflies_rotate = fireflies[:, :midpoint], fireflies[:, midpoint:]
        fireflies = torch.cat((fireflies, torch.roll(fireflies_rotate, shifts = 1, dims = (0,))), dim = 1)

    # decay exploration factor

    alpha *= alpha_decay

# print solution

fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

costs = cost_function(fireflies)
sorted_costs, sorted_indices = costs.sort(dim = -1)

fireflies = fireflies[sorted_indices]

print(f'best performing firefly for rosenbrock: {sorted_costs[0]:.3f}: {fireflies[0]}')
