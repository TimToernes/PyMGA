import os
import numpy as np
from scipy.spatial import ConvexHull
from ..utilities.general import solve_direcitons
from ..utilities.dask_helpers import start_dask_cluster


class MAA:
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

    def search_directions(self, n_samples, n_workers=4, max_iter=20):
        dim = self.dim
        dim_fullD = dim

        cluster, client = start_dask_cluster(workers=n_workers,
                                             try_slurm=False)

        max_n_dir = int(os.cpu_count())*20
        old_volume = 0
        epsilon = 1
        directions_searched = np.empty([0, dim])
        hull = None

        verticies = np.empty(shape=[0, dim])
        directions = np.empty((0, 0))
        sol_fullD = np.empty(shape=[0, dim_fullD])
        stat = np.empty(shape=[0])
        cost = np.empty(shape=[0])

        for i in range(max_iter):  # epsilon>MAA_convergence_tol:

            if len(verticies) <= 1:
                # logger.info('initializing directions')
                directions = np.concatenate([np.diag(np.ones(dim)),
                                             -np.diag(np.ones(dim))],
                                            axis=0)
            else:
                directions = -np.array(hull.equations)[:, 0:-1]
                if len(directions) > max_n_dir:
                    directions = directions[np.random.choice(len(directions),
                                                             max_n_dir)]

            if (len(directions)+len(directions_searched)) >= n_samples:
                remaining_evaluations = n_samples - len(directions_searched)
                directions = directions[:remaining_evaluations]

            directions_searched = np.concatenate([directions_searched,
                                                  directions],
                                                 axis=0)

            # Run all searches in parallel using DASK
            # logger.info(f'searching in {len(directions)} directions')
            verticies, sol_fullD, stat, cost = solve_direcitons(directions,
                                                                self.case,
                                                                client,
                                                                verticies,
                                                                sol_fullD,
                                                                stat,
                                                                cost)

            # logger.info('creating convex hull')
            try:
                hull = ConvexHull(verticies)
                # ,qhull_options='Qs C-1e-32')#,qhull_options='A-0.99')
            except Exception as e:
                print('did not manage to create hull first try')
                print(e)
                try:
                    hull = ConvexHull(verticies,
                                      qhull_options='Qx Q12 C-1e-32')
                except Exception as e:
                    print('did not manage to create hull second try')
                    print(e)

            delta_v = hull.volume - old_volume
            old_volume = hull.volume
            epsilon = delta_v/hull.volume

            print(f"""Itteration #{i},
                    total verticies {len(verticies)},
                    eps: {epsilon:.2f}""")

        return verticies, directions_searched, stat, cost
