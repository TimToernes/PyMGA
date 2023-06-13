import pandas as pd
import os


class Component:
    __validcomponents = {'Bus',
                         'Carrier',
                         'Generator',
                         'GlobalConstraint',
                         'Line',
                         'LineType',
                         'Link',
                         'Load',
                         'ShuntImpedance',
                         'StorageUnit',
                         'Store',
                         'SubNetwork',
                         'Transformer',
                         'TransformerType'}

    def __init__(self, name, attrs={'p_nom_opt'}):
        if name not in self.__validcomponents:
            raise AttributeError('Invalid name')

        self.name = name
        self.attrs = attrs

        for attr in self.attrs:
            setattr(self, attr, pd.DataFrame())

    def __repr__(self):
        return f'Component object representing {self.name}\'s'

    def append(self, network):

        for attr in self.attrs:
            list_name = network.components[self.name]['list_name']
            time_dependant = network.components[self.name]['attrs']
            time_dependant = time_dependant.loc[attr, 'varying']
            if time_dependant:
                list_name = list_name+'_t'
                network_comp = getattr(network, list_name)
                new_row = getattr(network_comp, attr).sum()
            else:
                network_comp = getattr(network, list_name)
                new_row = getattr(network_comp, attr)
            df = getattr(self, attr).append(new_row, ignore_index=True)
            setattr(self, attr, df)

    def save_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for attr in self.attrs:
            getattr(self, attr).to_csv(path+self.name+'-'+attr+'.csv')

    def from_csv(self, path):
        files = os.listdir(path)
        for f in files:
            if self.name in f:
                attr_name = f.split('-')[-1][:-4]
                setattr(self, attr_name, pd.read_csv(path+f, index_col=0))


class Solution:

    def __init__(self,
                 components={'Generator':  {'p_nom_opt', 'p'},
                             'Link': {'p_nom_opt',
                                      'p0',
                                      'p1',
                                      'p2',
                                      'p3',
                                      'p4'},
                             'Line': {'s_nom_opt', 'p0', 'p1'},
                             'StorageUnit': {'p_nom_opt', 'p'},
                             'Bus': {'marginal_price'},
                             'Store': {'e_nom_opt', 'p'}},
                 misc_info={'name',
                            'objective',
                            'path',
                            'chain',
                            'accepted',
                            'sample',
                            'global_constraints.constant.CO2Limit'}):

        self.components = components
        self.misc_info = misc_info

        for c in components:
            setattr(self, c, Component(c, attrs=self.components[c]))

        self.info = pd.DataFrame(columns=list(misc_info))

    def __repr__(self):
        n_solutions = self.info.shape[0]
        return f'Solutions object with {n_solutions} rows'

    def append_network(self, network):

        # Add compent variables
        for c_name in self.components:
            component = getattr(self, c_name)
            component.append(network)

        # add info variables
        new_row = {}
        for attr in self.misc_info:
            try:
                obj = network
                for s in attr.split('.'):
                    obj = getattr(obj, s)
                new_row[attr] = obj
            except AttributeError:
                new_row[attr] = None
        self.info = self.info.append(new_row, ignore_index=True)

    def append_solution(self, sol):
        # Appends an existing Solution class

        self.info = self.info.append(sol.info, ignore_index=True)

        # For all components, add all attributes
        for c_name in self.components:
            component_self = getattr(self, c_name)
            component_new = getattr(sol, c_name)
            for attr in component_self.attrs:
                df = getattr(component_self, attr)
                df = df.append(getattr(component_new, attr), ignore_index=True)
                setattr(component_self, attr, df)

    def save_csv(self, path):
        # Save all components
        if path[-1] != os.path.sep:
            path = path + os.path.sep

        for c_name in self.components:
            getattr(self, c_name).save_csv(path)

        # Save info df
        self.info.to_csv(path+'info.csv')

    @classmethod
    def from_network(cls, network):
        # Initializes the Solution class from a PyPSA network.nc file
        sol = cls()
        sol.append_network(network)
        return sol

    @classmethod
    def from_csv(cls, path):
        sol = cls()
        # read info df
        sol.info = pd.read_csv(path+'info.csv', index_col=0)

        # read componet df's
        for c_name in sol.components:
            getattr(sol, c_name).from_csv(path)
        return sol
