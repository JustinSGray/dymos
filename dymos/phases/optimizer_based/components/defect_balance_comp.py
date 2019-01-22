"""Define the BalanceComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import string_types, iteritems

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units

class DefectBalanceComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.

    Attributes
    ----------
    _state_vars : dict
        Cache the data provided during `add_balance`
        so everything can be saved until setup is called.
    """

    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')


    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(num_col_nodes,))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'f_computed': 'f_computed:{0}'.format(state_name),
                # 'defect': 'defects:{0}'.format(state_name),
            }

        for name, options in iteritems(state_options):


            self.add_output(name=name,
                            shape=(num_state_input_nodes, np.prod(options['shape'])),
                            units=options['units'])


            self.add_input(
                name=var_names['f_approx'],
                shape=(num_col_nodes,) + shape,
                desc='Estimated derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_col_nodes,) + shape,
                desc='Computed derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)


            # self.declare_partials(of=name, wrt=options['lhs_name'], rows=ar, cols=ar, val=1.0)
            # self.declare_partials(of=name, wrt=options['rhs_name'], rows=ar, cols=ar, val=1.0)

        # Setup partials
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            r = np.arange(num_col_nodes * size)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=r)

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_computed'],
                                  rows=r, cols=r)

            c = np.repeat(np.arange(num_col_nodes), size)
            self.declare_partials(of=var_names['defect'],
                                  wrt='dt_dstau',
                                  rows=r, cols=c)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each balance.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau']

        for state_name in state_options:
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            residuals[state_name] = ((f_approx - f_computed).T * dt_dstau).T

    def linearize(self, inputs, outputs, jacobian):
        dt_dstau = inputs['dt_dstau']
        for state_name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            partials[state_name, var_names['f_approx']] = k
            partials[state_name, var_names['f_computed']] = -k
            partials[state_name, 'dt_dstau'] = (f_approx - f_computed).ravel()



