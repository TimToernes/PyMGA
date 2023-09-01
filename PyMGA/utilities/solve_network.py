
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import gc
import os
import pypsa
from pypsa.descriptors import nominal_attrs
from pypsa.linopf import lookup, network_lopf, ilopf
from pypsa.linopt import (
    define_constraints,
    get_var,
    linexpr,
    write_bound,
    write_objective,
    join_exprs,
    get_dual,
    get_con,
    get_sol,
    define_variables,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import get_extendable_i, get_non_extendable_i
from pypsa.descriptors import free_output_series_dataframes

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

marginal_attr = {"Generator": "p", "Link": "p", "Store": "p", "StorageUnit": "p_dispatch"}


def patch_pyomo_tmpdir(tmpdir):
    # PYOMO should write its lp files into tmp here
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir

def prepare_network(n, solve_opts=None):
    if solve_opts is None:
        solve_opts = snakemake.config['solving']['options']

    if snakemake.config['foresight']=='myopic':
        add_land_use_constraint(n)

    return n


def add_land_use_constraint(n):
    if 'm' in snakemake.wildcards.clusters:
        # if generators clustering is lower than network clustering, land_use
        # accounting is at generators clusters
        for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
            existing_capacities = n.generators.loc[n.generators.carrier==carrier,"p_nom"]
            ind=list(set([i.split(sep=" ")[0] + ' ' + i.split(sep=" ")[1] for i in existing_capacities.index]))
            previous_years= [str(y) for y in 
                             snakemake.config["scenario"]["planning_horizons"] 
                             + snakemake.config["existing_capacities"]["grouping_years"]
                             if y < int(snakemake.wildcards.planning_horizons)]
            for p_year in previous_years:
                ind2 = [i for i in ind if  i + " " + carrier + "-" + p_year in existing_capacities.index]
                n.generators.loc[[i + " " + carrier + "-" + snakemake.wildcards.planning_horizons for i in ind2], "p_nom_max"] -= existing_capacities.loc[[i + " " + carrier + "-" + p_year for i in ind2]].rename(lambda x: x[:-4]+snakemake.wildcards.planning_horizons) 
    else:
        #warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'
        for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
            existing_capacities = n.generators.loc[n.generators.carrier==carrier,"p_nom"].groupby(n.generators.bus.map(n.buses.location)).sum()
            existing_capacities.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
            n.generators.loc[existing_capacities.index,"p_nom_max"] -= existing_capacities

    n.generators.p_nom_max[n.generators.p_nom_max<0]=0.


def assign_carriers(n):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    """

    if "Load" in n.carriers.index:
        n.carriers = n.carriers.drop("Load")

    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

    if n.links.empty:
        n.links["carrier"] = pd.Series(dtype=str)

    config = {
        "AC": {"color": "rosybrown", "nice_name": "HVAC Line"},
        "DC": {"color": "darkseagreen", "nice_name": "HVDC Link"},
    }
    for c in ["AC", "DC"]:
        if c in n.carriers.index:
            continue
        n.carriers = n.carriers.append(pd.Series(config[c], name=c))

def define_mga_constraint(n, sns, epsilon=None, with_fix=False):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    
    Build constraint defining near-optimal feasible space
    Parameters
    ----------
    n : pypsa.Network
    sns : Series|list-like
        snapshots
    epsilon : float, optional
        Allowed added cost compared to least-cost solution, by default None
    with_fix : bool, optional
        Calculation of allowed cost penalty should include cost of non-extendable components, by default None
    """

    if epsilon is None:
        epsilon = float(snakemake.wildcards.epsilon)

    if with_fix is None:
        with_fix = snakemake.config.get("include_non_extendable", True)

    expr = []

    # operation
    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(n.snapshot_weightings.loc[sns,'objective'], axis=0)
        )
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns])).stack())

    # investment
    for c, attr in nominal_attrs.items():
        cost = n.df(c)["capital_cost"][get_extendable_i(n, c)]
        if cost.empty:
            continue
        expr.append(linexpr((cost, get_var(n, c, attr)[cost.index])))

    lhs = pd.concat(expr).sum()

    if with_fix:
        ext_const = objective_constant(n, ext=True, nonext=False)
        nonext_const = objective_constant(n, ext=False, nonext=True)
        rhs = (1 + epsilon) * (n.objective_optimum + ext_const + nonext_const) - nonext_const
    else:
        ext_const = objective_constant(n)
        rhs = (1 + epsilon) * (n.objective_optimum + ext_const)

    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "mu_epsilon")

def objective_constant(n, ext=True, nonext=True):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    """

    if not (ext or nonext):
        return 0.0

    constant = 0.0
    for c, attr in nominal_attrs.items():
        i = pd.Index([])
        if ext:
            i = i.append(get_extendable_i(n, c))
        if nonext:
            i = i.append(get_non_extendable_i(n, c))
        constant += n.df(c)[attr][i] @ n.df(c).capital_cost[i]

    return constant

def define_mga_objective(n,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for dir_i,var_i in zip(direction,mga_variables):
        
        # Filter by carrier
        mask = n.df(var_i[0]).carrier.isin(var_i[1])
        # filter by country
        if len(var_i)>3: 
            mask = mask & n.df(var_i[0]).country.isin(var_i[3])

        model_vars = get_var(n,var_i[0],var_i[2])[mask]
        tmp_expr = linexpr((dir_i,model_vars)).sum()
        expr_list.append(tmp_expr)

    mga_obj = join_exprs(np.array(expr_list))
    write_objective(n,mga_obj)

def define_point_constraint(n,snapshots,point,options):
    scaling = 1
    
    mga_variables = options['mga_variables']
    for p_i,var_i in zip(point,mga_variables):
        # Filter by carrier
        mask = n.df(var_i[0]).carrier.isin(var_i[1])
        # filter by country
        if len(var_i)>3: 
            mask = mask & n.df(var_i[0]).country.isin(var_i[3])

        model_vars = get_var(n,var_i[0],var_i[2])[mask]
        expr = linexpr((scaling,model_vars)).sum()
        define_constraints(n,expr,'==',p_i*scaling,"custom",var_i[1][0])



# This function is adapted from 2022 Koen van Greevenbroek & Aleksander Grochowicz
# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
# SPDX-License-Identifier: GPL-3.0-or-later
def get_objective(n, sns):
    """Return the objective function as a linear expression.
    # This function is adapted from 2022 Koen van Greevenbroek & Aleksander Grochowicz
    # SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
    # SPDX-License-Identifier: GPL-3.0-or-later"""
    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective[
            sns.unique("period")
        ]

    if n._multi_invest:
        weighting = n.snapshot_weightings.objective.mul(period_weighting, level=0).loc[
            sns
        ]
    else:
        weighting = n.snapshot_weightings.objective.loc[sns]

    total = ""

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += cost @ n.df(c)[attr][ext_i]

    object_const = write_bound(n, constant, constant)
    total += linexpr((-1, object_const), as_pandas=False)[0]

    # marginal cost
    for c, attr in marginal_attr.items():
        cost = (
            get_as_dense(n, c, "marginal_cost", sns)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        terms = linexpr((cost, get_var(n, c, attr).loc[sns, cost.columns])).sum().sum()
        total += terms

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = get_var(n, c, attr).loc[ext_i]
        terms = linexpr((cost, caps)).sum()
        total += terms

    return total


def extra_functionality(n, snapshots, mga_options, direction, extra_func):
    
    # Add user defined constraints that were passed to pypsa_to_case
    if extra_func is not None:
        extra_func(n, snapshots)
    
    
    # Implement MGA constraint 
    if mga_options is not None:
        if mga_options['mga_slack'] is not None:
            
            # Get epsilon
            epsilon = mga_options['mga_slack']
            
            # Calculate MGA objective value
            max_obj = (1 + epsilon) * n.objective_optimum
            
            # Define constraint for near-optimal space
            define_constraints(n, get_objective(n, n.snapshots), "<=", max_obj, "GlobalConstraint", "Near_optimal")
            
            # Define system cost variable
            define_variables(n, 0, np.inf , 'system_cost','')
            
            # Define system cost adjusted constraint
            define_constraints(n, get_objective(n, n.snapshots) + linexpr((-1,get_var(n,'system_cost',''))),  '<=' , 0, 'cost_con')
            
            # Define mga objective via define_mga_objective function
            define_mga_objective(n, snapshots, direction, mga_options)
            
        else :
            define_point_constraint(n,snapshots,direction,mga_options)

def solve_network(n, extra_func, config=None, solver_log=None, opts=None, mga_options=None, direction=None, ):
    
    if config is None:
        config = snakemake.config['solving']
    solve_opts = config['options']

    solver_options = config['solver'].copy()
    if solver_log is None:
        solver_log = None
    solver_name = solver_options.pop('name')

    try:
        tmpdir = '/scratch/' + os.getenv('SLURM_JOB_ID') 
    except TypeError:
        tmpdir = 'tmp/'
        

    def run_lopf(n, allow_warning_status=False):
        free_output_series_dataframes(n)

        # Firing up solve will increase memory consumption tremendously, so
        # make sure we freed everything we can
        gc.collect()

        if mga_options is not None and mga_options['mga_slack'] is not None:
            skip_obj = True
        else : 
            skip_obj = False

        status, termination_condition = n.lopf(pyomo=False,
                                               solver_name=solver_name,
                                               solver_options=solver_options,
                                               solver_dir=tmpdir, 
                                               extra_functionality = lambda n,s: extra_functionality(n,s, mga_options, direction, extra_func),
                                               formulation=solve_opts['formulation'],
                                               keep_shadowprices='gas_limit',
                                               keep_references=True,
                                               skip_objective=skip_obj)


        if mga_options is not None and mga_options['mga_slack'] is not None:
            n.objective = float(get_sol(n,'system_cost'))

        assert status == "ok" or allow_warning_status and status == 'warning', \
            ("network_lopf did abort with status={} "
             "and termination_condition={}"
             .format(status, termination_condition))
            
        if status != "ok" or termination_condition != "optimal":
            raise ValueError(("network_lopf did abort with 'status = {}' and 'termination_condition = {}' "
             .format(status, termination_condition)))
            
        # assert termination_condition is 'optimal', \
        #     ("network_lopf did abort with 'status = {}' "
        #      "and 'termination_condition = {}'"
        #      .format(status, termination_condition))

        return status, termination_condition


    status, termination_condition = run_lopf(n, allow_warning_status=True)


    return n, status
