import torch

# test objective function - solution is close to all 1.'s

def rosenbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

# constants

steps = 1000
population_size = 100
dimensions = 4
lower_bound = -4.
upper_bound = 4.

beta0 = 2.           # exploitation factor, moving fireflies of low light intensity to high
gamma = 1.           # controls light intensity decay over distance
alpha = 0.2          # exploration factor
alpha_decay = 0.99   # exploration decay each step

# main algorithm

fireflies = torch.zeros((population_size, dimensions)).uniform_(lower_bound, upper_bound)

cost_function = rosenbrock

for _ in range(steps):

    # cost, which is inverse of light intensity

    costs = cost_function(fireflies)

    # fireflies with lower light intensity (high cost) moves towards the higher intensity (lower cost)

    move_mask = costs[:, None] > costs[None, :]

    # get vectors of fireflies to one another
    # calculate distance and the beta

    delta_positions = fireflies[None, :] - fireflies[:, None]

    distance = delta_positions.norm(dim = -1)

    betas = beta0 * (-gamma * distance ** 2).exp()

    random_walk = alpha * (torch.rand_like(fireflies) - 0.5) * (upper_bound - lower_bound)

    # move the fireflies

    fireflies += ((move_mask * betas)[..., None] * delta_positions).sum(dim = -2)

    fireflies += random_walk

    # decay exploration factor

    alpha *= alpha_decay

# print solution

costs = cost_function(fireflies)
sorted_costs, sorted_indices = costs.sort(dim = -1)

fireflies = fireflies[sorted_indices]

print(f'best performing firefly for rosenbrock: {sorted_costs[0]:.3f}: {fireflies[0]}')
