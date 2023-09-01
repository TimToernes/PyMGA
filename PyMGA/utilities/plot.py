# -*- coding: utf-8 -*-
"""
Created on 31/08/2023

@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
"""

def set_options():
    import matplotlib.pyplot as plt
    import matplotlib
    color_bg      = "0.99"          #Choose background color
    color_gridaxe = "0.85"          #Choose grid and spine color
    rc = {"axes.edgecolor":color_gridaxe} 
    plt.style.use(('ggplot', rc))           #Set style with extra spines
    plt.rcParams['figure.dpi'] = 300        #Set resolution
    plt.rcParams['figure.figsize'] = [10, 5]
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', titlesize=20)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']     #Change font to Computer Modern Sans Serif
    plt.rcParams['axes.unicode_minus'] = False          #Re-enable minus signs on axes))
    plt.rcParams['axes.facecolor']= color_bg             #Set plot background color
    plt.rcParams.update({"axes.grid" : True, "grid.color": color_gridaxe}) #Set grid color
    plt.rcParams['axes.grid'] = True
    # plt.fontname = "Computer Modern Serif"
    
def near_optimal_space_2D(all_variables, chosen_variables,
                          verticies, samples,
                          bins = 25, ax = None,
                          linewidth = 2, linecolor = 'gray',
                          xlim = [None, None], ylim = [None, None],
                          plot_MAA_points = False, filename = None, show_text = True,
                          textcolor = 'black',
                          title = 'Near-optimal space',
                          ):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import ConvexHull
    import numpy as np
    import pandas as pd
    '''
    Plots 2D slice of a near-optimal space based on two chosen variables.
    
    Required packages: 
        matplotlib
        scipy
        numpy
        pandas
    
    Args:
        all_variables (list): Boundary points as n-dimensional numpy 
                                    array of floats.
        n_samples (float): Amount of samples to draw from the hull
    
    Returns:
        sample
    '''
    
    set_options()
    
    # Dataframe with all verticies
    verticies_df = pd.DataFrame(verticies,
                                columns = all_variables)
    
    # Get verticies from only the variables specified
    variable_verticies = verticies_df[chosen_variables]
    
    variables = chosen_variables
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize = (10,8))
    
    if show_text:
        ax.set_xlabel(variables[0], fontsize = 24)
        ax.set_ylabel(variables[1], fontsize = 24)
        ax.set_title(title, color = textcolor)
    
    # Set x and y to be verticies for the first two variables
    x, y = variable_verticies[variables[0]], variable_verticies[variables[1]]
    
    samples_df = pd.DataFrame(samples,
                              columns = all_variables)
    
    # Set x and y as samples for this dimension
    x_samples = samples_df[variables[0]]
    y_samples = samples_df[variables[1]]
    
    # --------  Create 2D histogram --------------------
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples,
                                          bins = bins)
    
    # Create grid for pcolormesh
    x_grid, y_grid = np.meshgrid(xedges, yedges)
    
    # Create pcolormesh plot with square bins
    ax.pcolormesh(x_grid, y_grid, hist.T, cmap = 'Blues', 
                  zorder = 0)
    
    # Create patch to serve as hexbin label
    hb = mpatches.Patch(color = 'tab:blue')
    
    ax.grid('on')
    
    # --------  Plot hull --------------------
    hull = ConvexHull(variable_verticies.values)
    
    # plot simplexes
    for simplex in hull.simplices:
        l0, = ax.plot(variable_verticies.values[simplex, 0], variable_verticies.values[simplex, 1], 'k-', 
                color = linecolor, label = 'faces',
                linewidth = linewidth, zorder = 0)
    
    # list of legend handles and labels
    l_list, l_labels   = [l0, hb], ['Convex hull', 'Sample density']
    
    if plot_MAA_points:
        # Plot vertices from solutions
        l1, = ax.plot(x, y,
                  'o', label = "Near-optimal",
                  color = 'lightcoral', zorder = 2)
        l_list.append(l1)
        l_labels.append('MAA points')
        
    if show_text:
        ax.legend(l_list, l_labels, 
                  loc = 'center', ncol = len(l_list),
                  bbox_to_anchor = (0.5, -0.15), fancybox=False, shadow=False,)
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if not filename == None:
        fig.savefig(filename, format = 'pdf', bbox_inches='tight')
        
    return ax 
