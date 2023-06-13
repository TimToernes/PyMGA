import chaospy
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class DirectionSampler():
    """Class for drawign random directions on the unit hypersphere. 
    Using Chaospy to draw quassi-random samples from a 
    joint normal distribution
    """
    def __init__(self, dim, rule='halton'):
        self.count = 0
        self.dim = dim
        self.dist = self.create_dist()
        self.rule = rule

    def create_dist(self):
        # Creates a joint distribution with dim normal distributions. 
        DD = []
        for i in range(self.dim):
            DD.append(chaospy.chaospy.Normal(0, 1))
        distribution = chaospy.J(*DD)
        return distribution

    def draw_dir(self, n=1):
        s = self.dist.sample(n+self.count, rule=self.rule)[:, self.count:]
        norm = np.linalg.norm(s, axis=0)
        s_scaled = s/norm
        self.count += n
        return s_scaled.T


def solve_direcitons(directions,
                     case,
                     client,
                     verticies,
                     sol_fullD,
                     stat,
                     cost):
    """ Solve the model for the given directions
    directions: Directions to search in 
    case: test case object
    client: DASK client
    verticies: Known verticies
    """
    dim = directions.shape[1]
    variables = list(case.variables.keys())[:dim]
    params = (directions, [variables for i in range(len(directions))])
    n_solved = client.map(case.search_direction, *params, pure=False)
    
    res = client.gather(n_solved)
    for res_i in res:
        if res_i[2] == 'ok':
            verticies = np.append(verticies, np.array([res_i[0]]), axis=0)
            
            sol_fullD = np.append(sol_fullD, 
                                  np.array([list(res_i[1].values())]), 
                                  axis=0)
            stat = np.append(stat, np.array([res_i[2]]), axis=0)
            cost = np.append(cost, np.array([res_i[3]]), axis=0)
        else:
            print('Direction not solved with sucess')
    
    return verticies, sol_fullD, stat, cost


def check_large_volume(directions, verticies, sample, tol=0):
    """ Given a set of directions and verticies, compute
    wheater a point (sample) is inside
    all the hyperplanes defined by
    normals (directions) and points (verticies)
    """
    dist = []
    for i in range(len(verticies)):
        dist.append(directions[i]@(sample-verticies[i]))

    distances = np.array(dist)
    passing = np.all(distances >= -tol)

    if not passing:
        idx = list(np.where(~(distances >= -tol))[0])
        print(f'violating constraint {idx}')
        print(distances[idx])

    return passing


def calc_x0(directions, verticies):
    n = verticies.shape[0]
    s = np.random.rand(n-1)
    s.sort()
    s = np.insert(s, [0, n-1], [0, 1])
    s = np.diff(s)

    x0 = np.sum(verticies.T*s, axis=1)
    return x0


def check_small_volume(points_nD, p):
    # Form equality matrix
    A_eq = np.append(points_nD.T, [np.ones(points_nD.shape[0])], axis=0)
    A_eq = np.round(A_eq, 13)
    b_eq = np.append(p, [1])
    # Objective function
    c = np.zeros(points_nD.shape[0])

    m = gp.Model('interVol')
    x = m.addMVar(shape=points_nD.shape[0],
                  vtype=GRB.CONTINUOUS,
                  name="x",
                  lb=0,
                  ub=1)
    m.addConstr(A_eq @ x == b_eq, name="c")
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.setParam('LogToConsole', 0)
    m.optimize()
    m.Status

    return m.Status == 2
