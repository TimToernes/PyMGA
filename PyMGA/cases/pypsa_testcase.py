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


class PyPSA_case:
    def __init__(self,
                 config,
                 base_network_path,
                 variables=None,
                 tmp_network_path='tmp/networks/tmp.h5',
                 n_snapshots=100,
                 mga_slack=0.1):

        if variables is None:
            self.variables = {'x1': ['Generator',
                                    ['onwind', 'offwind-ac', 'offwind-dc'],
                                    'p_nom',
                                    ['NO', 'SE', 'FI', 'DK', 'GB', 'IE']],
                              'x2': ['Generator',
                                    ['onwind', 'offwind-ac', 'offwind-dc'],
                                    'p_nom',
                                    ['DE', 'AT', 'LU', 'BE', 'NL', 'CZ']],
                              'x3': ['Generator',
                                    ['onwind', 'offwind-ac', 'offwind-dc'],
                                    'p_nom',
                                    ['PL', 'LT', 'LV', 'EE', 'RO', 'SK', 'HU']],
                              'x4': ['Generator',
                                    ['onwind', 'offwind-ac', 'offwind-dc'],
                                    'p_nom',
                                    ['PT', 'ES', 'FR']],
                              'x5': ['Generator',
                                    ['onwind', 'offwind-ac', 'offwind-dc'],
                                    'p_nom',
                                    ['IT', 'CH', 'SI', 'GR', 'HR', 'RS', 'AL', 'ME', 'BA', 'MK', 'BG']],
                              'x6': ['Generator', ['solar'], 'p_nom', ['NO', 'SE', 'FI', 'DK', 'GB', 'IE']],
                              'x7': ['Generator', ['solar'], 'p_nom', ['DE', 'AT', 'LU', 'BE', 'NL', 'CZ']],
                              'x8': ['Generator',
                                    ['solar'],
                                    'p_nom',
                                    ['PL', 'LT', 'LV', 'EE', 'RO', 'SK', 'HU']],
                              'x9': ['Generator', ['solar'], 'p_nom', ['PT', 'ES', 'FR']],
                              'x10': ['Generator',
                                    ['solar'],
                                    'p_nom',
                                    ['IT', 'CH', 'SI', 'GR', 'HR', 'RS', 'AL', 'ME', 'BA', 'MK', 'BG']]}
        else:
            self.variables = variables

        self.base_network_path = base_network_path
        self.network_path = tmp_network_path
        self.config_path = 'networks/config.yaml'
        self.config = config

        self.n_snapshots = n_snapshots
        self.start_point = 0
        self.mga_slack = mga_slack
        self.objective_optimum = None

        self.write_network()

    def write_network(self):
        n = pypsa.Network(self.base_network_path)

        self.set_country(n)
        self.get_var_costs(n)

        n.snapshots = n.snapshots[self.start_point:self.start_point + self.n_snapshots]
        n.snapshot_weightings = n.snapshot_weightings[self.start_point:self.start_point+self.n_snapshots]
        n.snapshot_weightings = (n.snapshot_weightings*0 + int(8760/self.n_snapshots)).astype(int)

        # Write network to file
        for i in range(5):
            p = os.path.dirname(self.network_path)
            if not os.path.exists(p):
                # If it doesn't exist, create it
                os.makedirs(p)
            try:
                n.export_to_hdf5(self.network_path)
            except Exception as e:
                print('Error', e)
                pass

    def read_network(self):
        n = pypsa.Network()
        override_component_attrs = get_override_component_attrs()
        n = pypsa.Network(override_component_attrs=override_component_attrs)
        n.import_from_hdf5(self.network_path)
        self.set_country(n)
        n.objective_optimum = self.objective_optimum

        # with open(self.config_path) as f:
        #    config = yaml.safe_load(f)

        return n

    def solve(self):
        n = self.read_network()
        n, status = solve_network(n,
                                  config=self.config['solving'],
                                  )

        self.objective_optimum = n.objective

        all_variable_values = self.get_var_values(n)

        return n.objective, all_variable_values

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
                                  solver_log='logs/log.log')

        sol = Solution.from_network(n)
        if status == 'ok':
            all_variable_values = self.get_var_values(n)
        else:
            all_variable_values = {k: np.NaN for k in self.variables.keys()}

        spec_var_values = [all_variable_values.get(v) for v in variables]

        return spec_var_values, n.objective, sol


    def get_var_values(self, n, variables=None):
        if variables is None:
            variables = list(self.variables.keys())

        variable_cost = self.get_var_costs(n)

        variable_values = {}
        for var_i in self.variables:
            var_val = self.variables[var_i]
            df = n.df(var_val[0]).query('carrier==@var_val[1]')
            if len(var_val) > 3:
                df = df.query('country==@var_val[3]')
            val = df.loc[:, '{}_opt'.format(var_val[2])].sum()
            variable_values[var_i] = val

        try:
            variable_investment = {}
            for var_i in variable_cost:
                val = variable_values[var_i]*variable_cost[var_i]*1e-6
                variable_investment[var_i] = val
            
        except Exception as e:
            logger.error(e)
            logger.error(variable_cost) 
            logger.error(variable_values)
            logger.error(variables)
            raise Exception(e)

        return variable_values  # variable_investment

    def get_var_costs(self, n):
        variable_cost = {}

        for var_i in self.variables:
            var_val = self.variables[var_i]
            df = n.df(var_val[0]).query('carrier==@var_val[1]')
            if len(var_val) > 3:
                df = df.query('country==@var_val[3]')
            val = df.loc[:, 'capital_cost'].mean()
            variable_cost[var_i] = val

            self.variable_cost = variable_cost
        return variable_cost

    def set_country(self, n):
        # Add country name as column in component datasets
        for comp in n.one_port_components:
            df = n.df(comp)
            countries = [i[:2] for i in df.index]
            df['country'] = countries


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


def get_override_component_attrs():
    override_component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    override_component_attrs["Link"].loc["bus2"] = ["string", np.nan, np.nan, "2nd bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus3"] = ["string", np.nan, np.nan, "3rd bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus4"] = ["string", np.nan, np.nan, "4th bus", "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency2"] = ["static or series", "per unit", 1., "2nd bus efficiency", "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency3"] = ["static or series", "per unit", 1., "3rd bus efficiency", "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency4"] = ["static or series", "per unit", 1., "4th bus efficiency", "Input (optional)"]
    override_component_attrs["Link"].loc["p2"] = ["series", "MW", 0., "2nd bus output", "Output"]
    override_component_attrs["Link"].loc["p3"] = ["series", "MW", 0., "3rd bus output", "Output"]
    override_component_attrs["Link"].loc["p4"] = ["series", "MW", 0., "4th bus output", "Output"]
    return override_component_attrs
