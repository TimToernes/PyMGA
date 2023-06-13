import numpy as np
import os
from ..utilities.general import (solve_direcitons,
                                 DirectionSampler,
                                 check_large_volume,
                                 check_small_volume,
                                 calc_x0)
from ..utilities.dask_helpers import start_dask_cluster
from ..sampler.sampler import har_sample


class rMAA:
    def __init__(self, case):
        """
        """
        self.case = case
        self.dim = len(case.variables)

    def find_optimum(self):
        """ 
        """
        # Finding optimal solution
        self.obj, opt_sol = self.case.solve()
        self.opt_sol = list(opt_sol.values())[:self.dim]

        return self.opt_sol, self.obj
    
    def search_directions(self,
                          n_samples,
                          har_samples=5000,
                          n_workers=4,
                          max_iter=30,
                          tol=0.99):
        dim = self.dim
        dim_fullD = dim

        cluster, client = start_dask_cluster(workers=n_workers,
                                             try_slurm=False)

        stat = np.empty(shape=[0])
        cost = np.empty(shape=[0])
        directions = np.empty((0, 0))
        samples = None
        acc_small = None

        for i in range(max_iter):
            # Find directions
            if len(directions) == 0:
                print('initializing directions')
                new_directions = np.concatenate((np.diag(np.ones(dim)),
                                                -np.diag(np.ones(dim)),
                                                np.ones((1, dim)),
                                                -np.ones((1, dim))))
                directions = new_directions.copy()
                verticies = np.empty(shape=[0, dim])
                sol_fullD = np.empty(shape=[0, dim_fullD])
            else:
                print('searcing for new directions')
                max_iter = int(os.cpu_count())*4
                new_directions = find_new_directions(verticies,
                                                     directions,
                                                     samples,
                                                     acc_small,
                                                     client,
                                                     max_iter=max_iter)
                print(f'Found {len(new_directions)} new directions')

                if (len(directions)+len(new_directions)) > n_samples:
                    remaining_evaluations = n_samples-len(directions)
                    new_directions = new_directions[:remaining_evaluations]

                directions = np.append(directions, new_directions, axis=0)

            # Solve directions
            print(f'searching in {len(new_directions)} directions')
            verticies, sol_fullD, stat, cost = solve_direcitons(new_directions,
                                                                self.case,
                                                                client,
                                                                verticies,
                                                                sol_fullD,
                                                                stat,
                                                                cost)
            print('checking validity of samples')
            test = []
            for v in verticies:
                test.append(check_large_volume(directions,
                                               verticies,
                                               v,
                                               2000))
            violators = np.where(~np.array(test))[0]

            if len(violators) > 0:
                print(f'Deleting {len(violators)} violators')
                verticies = np.delete(verticies, violators, axis=0)
                directions = np.delete(directions, violators, axis=0)

            # Hit and run sample
            print(f'Hit and run sampling, {har_samples} samples')
            x0 = calc_x0(directions, verticies)
            samples = har_sample(har_samples, x0, directions, verticies)

            # Find acceptance rate
            acc_small = []
            for i in range(len(samples)):
                acc_small.append(check_small_volume(verticies, samples[i]))

            acc_rate = np.mean(acc_small)

            print(f"""Itteration #{i},
                   total verticies {len(directions)},
                   acceptance rate {acc_rate:.3f}""")

            if acc_rate > tol:
                break

            if len(directions) >= n_samples:
                print('Max function evaluations reached. Stopping')
                break
        return verticies, directions, stat, cost


def find_new_directions(verticies,
                        directions,
                        samples,
                        acc_small,
                        client,
                        max_iter=64,
                        min_count=1):
    """ Find new directions that result in the largest 
    number of rejected samples
    verticies: The verticies found with MAA method
    directions: Directions used in the MAA 
    samples: Hit-and-Run samples
    client: DASK client 
    max_iter: maximum number of iterations
    min_count: minimum number of rejected samples
    returns
    new_direction: set of new directions
    """
    # Initialize 
    updated_directions = directions
    new_directions = np.empty((0, directions.shape[1]))
    updated_verticies = verticies
    count = np.inf
    n_iter = 0
    # Samples that are between the two bounds
    samples_between = samples[~np.array(acc_small)]
    samples_remaining = samples_between

    while count > min_count and n_iter < max_iter:

        res = select_best_direction(updated_verticies,
                                    updated_directions,
                                    samples_remaining,
                                    client,
                                    n_new=2000,
                                    )
 
        best_dir, samples_remaining, count, v_best = res

        updated_directions = np.concatenate((updated_directions, [best_dir]))
        new_directions = np.concatenate((new_directions, [best_dir]))
        updated_verticies = np.concatenate((updated_verticies, [v_best]))

        print(f'Rejecting {count},samples, #iter {n_iter},max k {0:.2f}')
        n_iter += 1 

    return new_directions


def select_best_direction(verticies,
                          directions,
                          samples,
                          client,
                          n_new=500,
                          n_samples=5000):
    """ Find the direction resulting in the largest number of rejected samples
    verticies: The verticies found with MAA method
    directions: Directions used in the MAA 
    samples: Hit-and-Run samples
    client: DASK client 
    n_new: number of random directions to test
    returns
    best_dir: best direction
    samples_remaining: samples not rejected by new direction
    count: number of rejected samples
    """

    # Random directions
    dir_sampler = DirectionSampler(verticies.shape[1], rule='random')
    test_directions = dir_sampler.draw_dir(n_new)

    counts = []

    # Find the vertex, furthest in direction of the dir_i
    idx = np.argmax(-test_directions@verticies.T, axis=1)
    v = verticies[idx]
    # Compute the number of samples rejected for each direction
    for i in range(n_new):
        samples_rejected = sum(test_directions[i]@(samples-v[i]).T < 0)
        counts.append(samples_rejected)

    # Select the best direction
    idx_best_dir = np.argmax(counts)
    best_dir = test_directions[idx_best_dir]
    v_best = verticies[idx[idx_best_dir]]
    
    # Fin the remaining samples
    samples_remaining = samples[[best_dir@(s-v_best) > 0 for s in samples]]
    count = counts[idx_best_dir]
    
    return best_dir, samples_remaining, count, v_best
    