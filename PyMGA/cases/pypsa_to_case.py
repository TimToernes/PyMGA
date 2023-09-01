import pypsa
import numpy as np
from pypsa.descriptors import nominal_attrs
from pypsa.descriptors import get_extendable_i
import pandas as pd
import sys
import os
import warnings
import logging
from pypsa.linopt import get_sol
from .solutions import Solution
from ..utilities.solve_network import solve_network
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.ERROR)


def setup_applevel_logger(logger_name, file_name=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    msg = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(msg)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

logger = setup_applevel_logger('TestCase')


class PyPSA_to_case:
    '''
    A class that takes a pypsa networks and translates it into a case for the
    MGA/MAA/bMAA algorithms in the PyMGA package.
    '''
    def __init__(self,
                 config,
                 base_network_path,
                 extra_func = None,
                 variables=None,
                 tmp_network_path='tmp/networks/tmp.h5',
                 n_snapshots=8760,
                 mga_slack=0.1):
        # Initialize case object by setting some attributes and writing the 
        # network to disk.
        
        # Set attributes
        self.variables = variables
        self.base_network_path = base_network_path
        self.network_path = tmp_network_path
        self.config_path = 'networks/config.yaml'
        self.config = config
        self.extra_func = extra_func

        self.n_snapshots = n_snapshots
        self.start_point = 0
        self.mga_slack = mga_slack
        self.objective_optimum = None

        # Write network to disk
        self.write_network()


    def write_network(self):
        # Create network from given path or network
        n = pypsa.Network(self.base_network_path)
        
        # Adjust snapshot amount and weighting
        n.snapshots = n.snapshots[self.start_point:self.start_point + self.n_snapshots]
        n.snapshot_weightings = n.snapshot_weightings[self.start_point:self.start_point+self.n_snapshots]
        # n.snapshot_weightings = (n.snapshot_weightings*0 + int(8760/200)).astype(int)
        n.snapshot_weightings = (n.snapshot_weightings*0 + int(8760/self.n_snapshots)).astype(int)

        # Write network to file ---------------------
        p = os.path.dirname(self.network_path)
        if not os.path.exists(p):
            # If it doesn't exist, create it
            os.makedirs(p)
        try:
            # Export network
            n.export_to_hdf5(self.network_path)
        except Exception as e:
            print('Error', e)
            pass

    def read_network(self):
        # Create new network
        n = pypsa.Network()
        
        # Import network
        n.import_from_hdf5(self.network_path)
        
        # Set optimum objective value
        n.objective_optimum = self.objective_optimum

        return n

    def solve(self):
        # Get network
        n = self.read_network()
        
        # Solve network with options from config.yaml file

        
        n, status = solve_network(n,
                                  config=self.config['solving'],
                                  extra_func = self.extra_func
                                  )
            

        # Set objective optimum variable value
        self.objective_optimum = n.objective

        # Save variable values from network as variable
        all_variable_values = self.get_var_values(n)
        n_solved = n

        return n.objective, all_variable_values, n_solved

    def search_direction(self, direction, variables=None):
        if variables is None:
            variables = list(self.variables.keys())
            
        options = dict(mga_slack=self.mga_slack,
                       mga_variables=[self.variables[v] for v in variables])

        n = self.read_network()
        n, status = solve_network(n,
                                  mga_options=options,
                                  direction=direction,
                                  config=self.config['solving'],
                                  extra_func = self.extra_func,
                                  solver_log='logs/log.log')

        mga_variables = options['mga_variables']
        var_values = []
        for var_i in mga_variables:
            mask = n.df(var_i[0]).carrier.isin(var_i[1])
            if len(var_i) > 3: 
                mask = mask & n.df(var_i[0]).country.isin(var_i[3])

            var_values.append(get_sol(n, var_i[0], var_i[2])[mask].sum())

        if status == 'ok':
            all_variable_values = self.get_var_values(n)
        else:
            logger.debug(f'solver status {status}')
            all_variable_values = {k: np.NaN for k in self.variables.keys()}

        spec_var_values = [all_variable_values.get(v) for v in variables]

        return (spec_var_values,
                all_variable_values,
                status,
                n.objective,
                var_values)

    def solve_point(self, point, variables=None):
        if variables is None:
            variables = list(self.variables.keys())
            
        options = dict(mga_slack=None,
                       mga_variables=[self.variables[v] for v in variables])

        n = self.read_network()

        n, status = solve_network(n,
                                  mga_options=options,
                                  direction=point,
                                  config=self.config['solving'],
                                  extra_func = self.extra_func,
                                  solver_log='logs/log.log')

        sol = Solution.from_network(n)
        if status == 'ok':
            all_variable_values = self.get_var_values(n)
        else:
            all_variable_values = {k: np.NaN for k in self.variables.keys()}

        spec_var_values = [all_variable_values.get(v) for v in variables]

        return spec_var_values, n.objective, sol


    def get_var_values(self, n, variables=None):
        '''
        Get the value of each variable. 
        '''
        if variables is None:
            variables = list(self.variables.keys())

        variable_values = {}
        for var_i in self.variables:
            var_val = self.variables[var_i]
            df = n.df(var_val[0]).query('carrier==@var_val[1]')
            val = df.loc[:, '{}_opt'.format(var_val[2])].sum()
            variable_values[var_i] = val


        return variable_values  # variable_investment


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


def aggregate_costs(n, by_carrier=True):
    """
    Author: Fabian Neumann 
    Source: https://github.com/PyPSA/pypsa-eur-mga
    """

    assign_carriers(n)

    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
    )

    costs = {}
    for c in n.iterate_components(components.keys()):
        p_nom, p_attr = components[c.name]
        if c.df.empty:
            continue
        p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.pnl[p_attr].multiply(n.snapshot_weightings, axis=0).sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (
                (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
            )
    costs = pd.concat(costs) / 1e9  # bn EUR/a

    if by_carrier:
        costs = costs.groupby(level=2).sum()

    return costs


def calc_system_cost(self, network):
    # Cost
    capital_cost = sum(network.generators.p_nom_opt*network.generators.capital_cost) + sum(network.links.p_nom_opt*network.links.capital_cost) + sum(network.storage_units.p_nom_opt * network.storage_units.capital_cost)
    marginal_cost = network.generators_t.p.groupby(network.generators.carrier, axis=1).sum().sum() * network.generators.marginal_cost.groupby(network.generators.type).mean()
    total_system_cost = marginal_cost.sum() + capital_cost
    # total_system_cost = network.objective
    return total_system_cost


def annual_investment_cost(
    n: pypsa.Network, only_extendable: bool = False, use_opt: bool = False
) -> float:
    """Compute the annual investment costs in a PyPSA network.
    This function assumes that capital costs in `n` are proportional to
    the length of time that `n` is defined over. Hence, the total
    investment costs in `n` are scaled down by the number of years `n`
    is defined over in order to get an annual figure.
    If `only_extendable` is set, only include extendable technologies in the
    calculation.
    If `use_opt` is set, use the *_nom_opt capacities instead of the
    *_nom capacities.
    """
    total = 0
    for c, attr in nominal_attrs.items():
        i = get_extendable_i(n, c) if only_extendable else n.df(c).index
        v = attr + "_opt" if use_opt else attr
        total += (n.df(c).loc[i, "capital_cost"] * n.df(c).loc[i, v]).sum()
    # Divide by the number of years the network is defined over.
    # Disregard leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return total


def annual_variable_cost(
    n: pypsa.Network,
) -> float:
    """Compute the annual variable costs in a PyPSA network `n`."""
    weighting = n.snapshot_weightings.objective
    total = 0
    # Add variable costs for generators
    total += (
        n.generators_t.p[n.generators.index].multiply(weighting, axis=0).sum(axis=0)
        * n.generators.marginal_cost
    ).sum()
    # Add variable costs for links (lines have none), 
    # in our model all 0 though.
    total += (
        n.links_t.p0[n.links.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.links.marginal_cost
    ).sum()
    # Add variable costs for stores
    total += (
        n.stores_t.p[n.stores.index].abs().multiply(weighting, axis=0).sum(axis=0)
        * n.stores.marginal_cost
    ).sum()
    # Add variable costs for storage units
    total += (
        n.storage_units_t.p[n.storage_units.index]
        .abs()
        .multiply(weighting, axis=0)
        .sum(axis=0)
        * n.storage_units.marginal_cost
    ).sum()
    # Divide by the number of years the network is defined over. Disregard
    # leap years.
    total /= n.snapshot_weightings.objective.sum() / 8760
    return 

