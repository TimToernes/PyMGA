import numpy as np
from ..utilities.general import check_large_volume, calc_x0, DirectionSampler


def har_sample(n_samples, x0, directions, verticies):
    """ Hit-and-Run sampler
    Sample n_samples starting from x0
    n_samples: number of samples to draw
    x0: starting point
    directions: Directions that have
    been searched in with the MAA algorithm phase 1
    vertivies: The verticies found by searing in
    directions with the MAA algorithm
    """
    for i in range(10):
        if not check_large_volume(directions, verticies, x0, tol=1000):
            print(f'x0 not in large volume. Trying again. i:{i}')
            x0 = calc_x0(directions, verticies)
        else:
            break
    # We want to represent the bounding hyperplanes as
    # normalvectors and offsets.
    # Offset is the distance from origo to the
    # plane in the normal direction.
    offsets = [directions[i]@(-verticies[i]) for i in range(len(directions))]
    A = -directions  # Matrix containing all normal vectors
    b = offsets  # vector of offsets

    # Initialize parameters
    x_i = x0
    samples = np.empty((n_samples, len(x0)))
    dir_sampler = DirectionSampler(len(x0))
    directions = dir_sampler.draw_dir(n_samples)
    for j in range(n_samples):
        direct_i = directions[j]
        # Distances from x_i to bounding planes in direction direct_i
        t_range = (b-A@x_i)/(A@direct_i)
        filt_max = A@direct_i > 0
        filt_min = A@direct_i < 0
        lambda_max = min(t_range[filt_max])  # Maximum stepsize
        lambda_min = max(t_range[filt_min])  # Minimum stepsize
        lambda_i = np.random.uniform(lambda_min, lambda_max, 1)
        x_new = x_i+direct_i*lambda_i
        samples[j, :] = x_new
        x_i = x_new

    return samples
