# -*- coding: utf-8 -*-
"""
Created on 29/8/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
    
Description:
    Example case based on the North Sea Energy Island. Consists of an island 
    with wind, P2X and storage capacity, connected to several countries.
    This example include custom constraints being defined using extra_func.
"""

import os
import sys

# Add parent folder to directory to load PyMGA package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import PyMGA
from PyMGA.utilities.plot import near_optimal_space_2D
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pypsa_netview.draw import draw_network
import pandas as pd
from ttictoc import tic, toc


if __name__ == '__main__':
    
    # Create or load network
    network = 'island_example_network.nc'
    
    # Define total island area
    total_area = 0.5*120_000 #[m^2]
    
    # Define area uses
    area_use = pd.Series( data = {'storage':  01.0,  #[m^2/MWh] Capacity
                                  'hydrogen': 02.1,  #[m^2/MW] capacity
                                  'data':     27.3,  #[m^2/MW] IT output
                                  })
    
    
    # Load options from configuration file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
        
    # Set MAA variables to explore
    variables = {'x1': ['Generator',
                        ['Data'],
                        'p_nom',],
                'x2': ['Generator',
                        ['P2X'],
                        'p_nom',],
                'x3': ['Store',
                        ['Storage'],
                        'e_nom',]
                        } 
    
    # variables = {'x1': ['Link',
    #                     ['Island_to_Denmark'],
    #                     'p_nom',],
    #             'x2': ['Link',
    #                     ['Island_to_Norway'],
    #                     'p_nom',],
    #             'x3': ['Link',
    #                     ['Island_to_Belgium'],
    #                     'p_nom',]
    #                     } 
    
    
    # Define constraints to be passed to extra_functionalities in n.lopf()
    def extra_func(n, snapshots):
        
        ### Define custom constraints
        # Define total link capacity constraint
        def link_constraint(n):
            # Create a constraint that limits the sum of link capacities
            from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
            
            # Get link info from network
            link_names = ['Island_to_Denmark', 'Island_to_Norway', 'Island_to_Germany',
                          'Island_to_Netherlands', 'Island_to_Belgium',
                          'Island_to_United Kingdom']               # List of main link names
            link_t     = 3_000           # Maximum total link capacity
            
            # Get all link variables, and filter for only main link variables
            vars_links   = get_var(n, 'Link', 'p_nom')
            vars_links   = vars_links[link_names]
            
            # Sum up link capacities of chosen links (lhs), and set limit (rhs)
            rhs          = link_t
            lhs          = join_exprs(linexpr((1, vars_links)))
            
            #Define constraint and name it 'Total constraint'
            define_constraints(n, lhs, '<=', rhs, 'Link', 'Sum constraint')
          
        # Define constraint forcing links to/from a country to have same capacity
        def marry_links(n):
            from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
            
            vars_links   = get_var(n, 'Link', 'p_nom')
            
            if not hasattr(n, 'connected_countries'):
                n.connected_countries =  [
                                        "Denmark",         
                                        "Norway",          
                                        "Germany",         
                                        "Netherlands",     
                                        "Belgium",         
                                        "United Kingdom"
                                        ]
            
            for country in n.connected_countries:
                
                lhs = linexpr((1, vars_links['Island_to_' + country]),
                              (-1, vars_links[country + '_to_Island']))
                
                define_constraints(n, lhs, '=', 0, 'Link', country + '_link_capacity_constraint')
        
        # Define area constraint affecting P2X, Data and Storage on island
        def area_constraint(n):
            # Get variables for all generators and store
            from pypsa.linopt import get_var, linexpr, define_constraints
            
            # Get variables to include in constraint
            vars_gen   = get_var(n, 'Generator', 'p_nom')
            vars_store = get_var(n, 'Store', 'e_nom')
            
            # Apply area use on variable and create linear expression 
            lhs = linexpr(
                           (area_use['hydrogen'], vars_gen["P2X"]), 
                           (area_use['data'],     vars_gen["Data"]), 
                           (area_use['storage'],  vars_store['Storage'])
                          )
            
            # Define area use limit
            rhs = total_area #[m^2]
            
            # Define constraint
            define_constraints(n, lhs, '<=', rhs, 'Island', 'Area_Use')
        
        ### Call custom constraints 
        link_constraint(n)
        marry_links(n)
        area_constraint(n)



    #### PyMGA ####
    # PyMGA: Build case from PyPSA network
    case = PyMGA.cases.PyPSA_to_case(config, 
                                     network,
                                     extra_func = extra_func,
                                     variables = variables,
                                     mga_slack = 0.1,
                                     n_snapshots = 8760)
    
    # PyMGA: Choose MAA method
    method = PyMGA.methods.MAA(case)
    
    # PyMGA: Solve optimal system
    opt_sol, obj, n_solved = method.find_optimum()
    
    # Draw optimal system (optional)
    draw_network(n_solved, show_capacities = True)
    
    # PyMGA: Search near-optimal space using chosen method
    tic()
    verticies, directions, _, _ = method.search_directions(7, n_workers = 16)
    print(toc())

    # PyMGA: Sample the identified near-optimal space
    MAA_samples = PyMGA.sampler.har_sample(100_000, x0 = np.zeros(len(variables.keys())), 
                                           directions = directions, 
                                           verticies = verticies)

#
    #### Processing results ####
    # Plot near-optimal space of Data and P2X
    all_variables    = list(variables.keys())
    chosen_variables = ['x1', 'x2']
    near_optimal_space_2D(all_variables, chosen_variables,
                          verticies, MAA_samples,
                          plot_MAA_points = True,)
