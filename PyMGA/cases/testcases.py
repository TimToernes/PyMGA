import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import scipy
import itertools
import warnings
import logging
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.ERROR)

#import pandas as pd
#import os
#import sys
#from pathlib import Path

#from mga_functions import define_mga_constraint, define_mga_objective, aggregate_costs
#from solve_network import solve_network
#from solutions import Solution



#%%

class SyntheticCase():
    """ 
    Parent class for the synthetic cases 
    The class contains shared functions 
    """
    def __init__(self, dim):
        self.variables = {'x'+str(i):'x'+str(i) for i in range(dim)}
        self.A = None
        self.b = None
        self.dim = dim

    def solve(self):
        d = np.ones(self.dim)
        x = self._model(d)
        sol = {var:sol for var,sol in zip(self.variables.keys(),x)}
        return 0,sol

    def search_direction(self,d,vars):
        d = -np.array(d)
        x = self._model(d)
        sol = {var:sol for var,sol in zip(self.variables.keys(),x)}
        return x,sol,'ok',0

    def _draw_dir(self,dim,seed=None):
        s = np.random.normal(0,1,dim)
        return s/np.linalg.norm(s)
    
    def _model(self,d):
        A = self.A
        b = self.b    
        m = gp.Model('model')
        x = m.addMVar(shape=A.shape[1], name="x",lb=-1,ub=1)
        m.addConstr(A @ x <= b, name="c")
        m.setObjective(-d @ x, GRB.MINIMIZE)
        m.setParam('LogToConsole',1)
        m.optimize()
        return x.X
    
    def plot(self):
        A = self.A
        b = self.b 
        for d_i,o_i in zip(A,b):
            o_i = o_i/np.linalg.norm(d_i)
            d_i = d_i/np.linalg.norm(d_i)
            
            plt.quiver((d_i*o_i)[0],(d_i*o_i)[1],-d_i[1],d_i[0],scale=1,width=0.001)
            plt.quiver((d_i*o_i)[0],(d_i*o_i)[1],d_i[1],-d_i[0],scale=1,width=0.001)
            

        #plt.scatter(samples[:,0],samples[:,1],s=8,alpha=.5)
        plt.axis('equal')
    


class Cube(SyntheticCase):
    """
    Test case consiting of a hypercube that is cut with a series of hyperplanes
    """
    def __init__(self,dim,cuts,max_dist=0.9,random_length=False):
        self.dim = dim
        self.variables = {'x'+str(i):'x'+str(i) for i in range(dim)}
        self.cuts = cuts
        self.random_length = random_length
        self.max_dist = max_dist

        self.A = np.concatenate((np.diag(np.ones(dim)),-np.diag(np.ones(dim))))
        self.b = np.ones(2*dim)

        np.random.seed(42)
        for i in range(cuts):
            self._add_cut(max_dist=self.max_dist)

        self.mga_slack = 0
        self.n_snapshots = 0
        self.base_network_path = "none"
        

    def volume(self):
        return 2**self.dim

    def _add_cut(self,max_dist=0.8):
        d = self._draw_dir(self.dim)
        if self.random_length:
            offset = np.random.uniform(.5,1)
        else :
            offset = 1
        self.A = np.append(self.A,[d],axis=0)
        self.b = np.append(self.b,[offset],axis=0)


class CubeCorr(SyntheticCase):
    """
    Test case consiting of a hypercube that is cut with a series of hyperplanes
    """
    def __init__(self,dim,margin=0.7):
        self.dim = dim
        self.variables = {'x'+str(i):'x'+str(i) for i in range(dim)}

        #self.A = np.concatenate((-np.diag(np.ones(dim)),[np.ones(dim)]))
        #self.b = np.concatenate((np.zeros(dim),[np.sqrt(dim)]))
        np.random.seed(42)
        A = np.zeros((int(2*scipy.special.binom(dim,2)),dim))
        b = np.zeros((int(2*scipy.special.binom(dim,2))))
        for i,pair in enumerate(itertools.combinations(range(dim), 2)):
            offset = int(scipy.special.binom(dim,2))
            A[i,pair[0]] = 1
            A[i,pair[1]] = 1
            A[offset+i,pair[0]] = 1
            A[offset+i,pair[1]] = -1

            coef = np.random.uniform(0,np.sqrt(2))
        
            

            b[i] = coef+margin
            b[i+offset] = np.sqrt(dim)-coef+margin


        self.A = np.concatenate((A,-A))
        self.b = np.concatenate((b,b))

        self.A = np.concatenate((self.A,np.diag(np.ones(dim)),-np.diag(np.ones(dim))))
        self.b = np.concatenate((self.b,np.ones(dim),np.ones(dim)))


        self.mga_slack = 0
        self.n_snapshots = 0
        self.base_network_path = "none"

    def volume(self):
        return 2**self.dim


class CrossPoly(SyntheticCase):
    """
    Test case representing a convex polytope. 
    The polytope is the interesection of a hyper-cube with width 2 (-1,1), and a cross-polytope 
    The width of the cross polytope is changed depending on number of dimensions to keep the volume within the 
    same order of magnitude, despite changing dimensions. 
    """
    def __init__(self,dim):
        self.dim = dim
        self.variables = {'x'+str(i):'x'+str(i) for i in range(dim)}
        #width = 1.1*2
        cut_point = .7
        self.width =  2*np.sqrt(2*(cut_point * np.sqrt(self.dim))**2)
        self.verticies = 2**dim*dim
        
        A = np.concatenate((np.diag(np.ones(dim)),np.diag(-np.ones(dim))))
        self.A = np.append(A,np.array(list(itertools.product([1,-1],repeat=dim))),axis=0)
        self.b = np.concatenate((np.ones(dim*2),np.ones(2**dim)*self.width/2))

        self.mga_slack = 0
        self.n_snapshots = 0
        self.base_network_path = "none"
     
    def volume(self):
        width = self.width
        dim = self.dim
        vol = ((width)**dim)/np.math.factorial(dim) - dim * ((width-2)**dim)/np.math.factorial(dim)
        return vol

    
