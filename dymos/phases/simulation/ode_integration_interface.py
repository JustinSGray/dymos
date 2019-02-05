from __future__ import print_function, division, absolute_import

import sys
from collections import OrderedDict

import numpy as np
from dymos.utils.simulation.components.control_interpolation_comp import ControlInterpolationComp
from dymos.utils.simulation.components.state_rate_collector_comp import StateRateCollectorComp
from dymos.utils.simulation.progress_bar_observer import ProgressBarObserver
from dymos.utils.simulation.phase_simulation_results import PhaseSimulationResults
from dymos.utils.simulation.std_out_observer import StdOutObserver
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.problem import Problem
from scipy.integrate import ode
from six import iteritems


class ODEIntegrationInterface(object):
    """
    Given a system class, create a callable object with the same signature as that required
    by scipy.integrate.ode::

        f(t, x, *args)

    Internally, this is accomplished by constructing an OpenMDAO problem using the ODE with
    a single node.  The interface populates the values of the time, states, and controls,
    and then calls `run_model()` on the problem.  The state rates generated by the ODE
    are then returned back to scipy ode, which continues the integration.

    Parameters
    ----------
    phase_name : str
        The name of the phase being simulated.
    ode_class : class
        The ODEClass belonging to the phase being simulated.
    time_options : dict of {str: TimeOptionsDictionary}
        The time options for the phase being simulated.
    state_options : dict of {str: StateOptionsDictionary}
        The state options for the phase being simulated.
    control_options : dict of {str: ControlOptionsDictionary}
        The control options for the phase being simulated.
    design_parameter_options : dict of {str: DesignParameterOptionsDictionary}
        The design parameter options for the phase being simulated.
    input_parameter_options : dict of {str: InputParameterOptionsDictionary}
        The input parameter options for the phase being simulated.
    ode_init_kwargs : dict
        Keyword argument dictionary passed to the ODE at initialization.
    """
    def __init__(self, phase_name, ode_class, time_options, state_options, control_options,
                 design_parameter_options, input_parameter_options,
                 ode_init_kwargs=None):

        self.phase_name = phase_name

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = OrderedDict()
        self.time_options = time_options
        self.control_options = control_options
        self.design_parameter_options = design_parameter_options
        self.input_parameter_options = input_parameter_options
        self.control_interpolants = {}

        # Stored initial value of time so that time_phase can be calculated
        self.t_initial = 0.0

        pos = 0

        for state, options in iteritems(state_options):
            self.state_options[state] = {'rate_source': options['rate_source'],
                                         'pos': pos,
                                         'shape': options['shape'],
                                         'size': np.prod(options['shape']),
                                         'units': options['units'],
                                         'targets': options['targets']}
            pos += self.state_options[state]['size']

        self._state_vec = np.zeros(pos, dtype=float)
        self._state_rate_vec = np.zeros(pos, dtype=float)

        #
        # Build odeint problem interface
        #
        self.prob = Problem(model=Group())
        model = self.prob.model

        # The time IVC
        ivc = IndepVarComp()
        time_units = ode_class.ode_options._time_options['units']
        ivc.add_output('time', val=0.0, units=time_units)
        ivc.add_output('time_phase', val=0.0, units=time_units)
        ivc.add_output('t_initial', val=0.0, units=time_units)
        ivc.add_output('t_duration', val=0.0, units=time_units)

        model.add_subsystem('time_input', ivc, promotes_outputs=['*'])

        model.connect('time', ['ode.{0}'.format(tgt) for tgt in
                               ode_class.ode_options._time_options['targets']])

        model.connect('time_phase', ['ode.{0}'.format(tgt) for tgt in
                                     ode_class.ode_options._time_options['time_phase_targets']])

        # self.prob.model.connect('t_initial',
        #                         ['ode.{0}'.format(tgt) for tgt in
        #                          self.ode_options._time_options['t_initial_targets']])
        #
        # self.prob.model.connect('t_duration',
        #                         ['ode.{0}'.format(tgt) for tgt in
        #                          self.ode_options._time_options['t_duration_targets']])

        # The States Comp
        for name, options in iteritems(self.state_options):
            ivc.add_output('states:{0}'.format(name),
                           shape=(1, np.prod(options['shape'])),
                           units=options['units'])

            rate_src = self._get_rate_source_path(name)

            model.connect(rate_src,
                          'state_rate_collector.state_rates_in:{0}_rate'.format(name))

            if options['targets'] is not None:
                model.connect('states:{0}'.format(name),
                              ['ode.{0}'.format(tgt) for tgt in options['targets']])

        if self.control_options:
            self._interp_comp = ControlInterpolationComp(time_units=time_units,
                                                         control_options=control_options)
            self._interp_comp.interpolants = self.control_interpolants

            model.add_subsystem('indep_controls', self._interp_comp, promotes_outputs=['*'])

            model.connect('time', ['indep_controls.time'])

            for name, options in iteritems(self.control_options):
                if name in ode_class.ode_options._parameters:
                    targets = ode_class.ode_options._parameters[name]['targets']
                    model.connect('controls:{0}'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in targets])
                if options['rate_param']:
                    rate_param = options['rate_param']
                    rate_targets = ode_class.ode_options._parameters[rate_param]['targets']
                    model.connect('control_rates:{0}_rate'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in rate_targets])
                if options['rate2_param']:
                    rate2_param = options['rate2_param']
                    rate2_targets = ode_class.ode_options._parameters[rate2_param]['targets']
                    model.connect('control_rates:{0}_rate2'.format(name),
                                  ['ode.{0}'.format(tgt) for tgt in rate2_targets])

        for name, options in iteritems(self.design_parameter_options):
            ivc.add_output('design_parameters:{0}'.format(name),
                           shape=np.prod(options['shape']),
                           units=options['units'])
            if options['targets'] is not None:
                model.connect('design_parameters:{0}'.format(name),
                              ['ode.{0}'.format(tgt) for tgt in options['targets']])

        for name, options in iteritems(self.input_parameter_options):
            ivc.add_output('input_parameters:{0}'.format(name),
                           shape=options['shape'],
                           units=options['units'])
            if options['targets'] is not None:
                model.connect('input_parameters:{0}'.format(name),
                              ['ode.{0}'.format(tgt) for tgt in options['targets']])

        # The ODE System
        model.add_subsystem('ode', subsys=ode_class(num_nodes=1, **ode_init_kwargs))

        # The state rate collector comp
        self.prob.model.add_subsystem('state_rate_collector',
                                      StateRateCollectorComp(state_options=self.state_options,
                                                             time_units=time_options['units']))

        # Flag that is set to true if has_controls is called
        self._has_dynamic_controls = False

    def _get_rate_source_path(self, state_var):
        var = self.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            rate_path = 'time_phase'
        elif var in self.state_options:
            rate_path = 'states:{0}'.format(var)
        elif var in self.control_options:
            rate_path = 'controls:{0}'.format(var)
        elif var in self.design_parameter_options:
            rate_path = 'design_parameters:{0}'.format(var)
        elif var in self.input_parameter_options:
            rate_path = 'input_parameters:{0}'.format(var)
        elif var.endswith('_rate') and var[:-5] in self.control_options:
            rate_path = 'control_rates:{0}'.format(var)
        elif var.endswith('_rate2') and var[:-6] in self.control_options:
            rate_path = 'control_rates:{0}'.format(var)
        else:
            rate_path = 'ode.{0}'.format(var)

        return rate_path

    # def setup(self, check=False):
    #     """
    #     Call setup on the ScipyODEIntegrator's problem instance.
    #
    #     Parameters
    #     ----------
    #     check : bool
    #         If True, run setup on the problem instance with checks enabled.
    #         Default is False.
    #
    #     """
    #     model = self.prob.model
    #
    #     order = ['time_input', 'indep_states']
    #
    #     if self.control_options:
    #         model.add_subsystem('indep_controls', self._interp_comp, promotes_outputs=['*'])
    #
    #         order += ['indep_controls']
    #         model.connect('time', ['indep_controls.time'])
    #
    #         for name, options in iteritems(self.control_options):
    #             if name in self.ode_options._parameters:
    #                 targets = self.ode_options._parameters[name]['targets']
    #                 model.connect('controls:{0}'.format(name),
    #                               ['ode.{0}'.format(tgt) for tgt in targets])
    #             if options['rate_param']:
    #                 rate_param = options['rate_param']
    #                 rate_targets = self.ode_options._parameters[rate_param]['targets']
    #                 model.connect('control_rates:{0}_rate'.format(name),
    #                               ['ode.{0}'.format(tgt) for tgt in rate_targets])
    #             if options['rate2_param']:
    #                 rate2_param = options['rate2_param']
    #                 rate2_targets = self.ode_options._parameters[rate2_param]['targets']
    #                 model.connect('control_rates:{0}_rate2'.format(name),
    #                               ['ode.{0}'.format(tgt) for tgt in rate2_targets])
    #
    #     if self.design_parameter_options:
    #         ivc = model.add_subsystem('design_params', IndepVarComp(), promotes_outputs=['*'])
    #
    #         order += ['design_params']
    #
    #         for name, options in iteritems(self.design_parameter_options):
    #             ivc.add_output('design_parameters:{0}'.format(name), val=np.zeros(options['shape']),
    #                            units=options['units'])
    #             if name in self.ode_options._parameters:
    #                 targets = self.ode_options._parameters[name]['targets']
    #                 model.connect('design_parameters:{0}'.format(name),
    #                               ['ode.{0}'.format(tgt) for tgt in targets])
    #
    #     if self.input_parameter_options:
    #         ivc = model.add_subsystem('input_params', IndepVarComp(), promotes_outputs=['*'])
    #
    #         order += ['input_params']
    #
    #         for name, options in iteritems(self.input_parameter_options):
    #             ivc.add_output('input_parameters:{0}'.format(name), val=np.zeros(options['shape']),
    #                            units=options['units'])
    #             if options['target_param'] in self.ode_options._parameters:
    #                 targets = self.ode_options._parameters[options['target_param']]['targets']
    #                 model.connect('input_parameters:{0}'.format(name),
    #                               ['ode.{0}'.format(tgt) for tgt in targets])
    #
    #     for name, options in iteritems(self.state_options):
    #         self.prob.model.connect(self._get_rate_source_path(name),
    #                                 'state_rate_collector.state_rates_in:{0}_rate'.format(name))
    #
    #     order += ['ode', 'state_rate_collector']
    #     model.set_order(order)
    #
    #     self.prob.setup(check=check)

    # def set_interpolant(self, name, interpolant):
    #     """
    #     Set the interpolator to be used for the control of the given name.
    #
    #     Parameters
    #     ----------
    #     name : str
    #         The name of the control whose interpolant is being specified.
    #     interpolant : object
    #         An object that provides interpolation for the control as a function of time.
    #         The object must have methods `eval(t)` which returns the interpolated value
    #         of the control at time t, and `eval_deriv(t)` which returns the interpolated
    #         value of the first time-derivative of the control at time t.
    #     """
    #     self._interp_comp.interpolants[name] = interpolant

    # def set_design_param_value(self, name, val, units=None):
    #     """
    #     Sets the values of design parameters in the problem, once self.prob has been setup.
    #
    #     Parameters
    #     ----------
    #     name : str
    #         The name of the design parameter whose value is being set
    #     val : float or ndarray
    #         The value of the design parameter
    #     units : str or None
    #         The units in which the design parameter value is set.
    #
    #     """
    #     self.prob.set_val('design_parameters:{0}'.format(name), val, units)
    #
    # def set_input_param_value(self, name, val, units=None):
    #     """
    #     Sets the values of input parameters in the problem, once self.prob has been setup.
    #
    #     Parameters
    #     ----------
    #     name : str
    #         The name of the design parameter whose value is being set
    #     val : float or ndarray
    #         The value of the design parameter
    #     units : str or None
    #         The units in which the design parameter value is set.
    #
    #     """
    #     self.prob.set_val('input_parameters:{0}'.format(name), val, units)

    def _unpack_state_vec(self, x):
        """
        Given the state vector in 1D, extract the values corresponding to
        each state into the ode integrators problem states.

        Parameters
        ----------
        x : np.array
            The 1D state vector.

        Returns
        -------
        None

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self.prob['states:{0}'.format(state_name)][0, ...] = x[pos:pos + size]

    def _pack_state_rate_vec(self):
        """
        Pack the state rates into a 1D vector for use by scipy odeint.

        Returns
        -------
        dXdt: np.array
            The 1D state-rate vector.

        """
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self._state_rate_vec[pos:pos + size] = \
                np.ravel(self.prob['state_rate_collector.'
                                   'state_rates:{0}_rate'.format(state_name)])
        return self._state_rate_vec

    def _pack_state_vec(self, x_dict):
        """
        Pack the state into a 1D vector for use by scipy.integrate.ode.

        Returns
        -------
        x: np.array
            The 1D state vector.

        """
        self._state_vec[:] = 0.0
        for state_name, state_options in self.state_options.items():
            pos = state_options['pos']
            size = state_options['size']
            self._state_vec[pos:pos + size] = np.ravel(x_dict[state_name])
        return self._state_vec

    def __call__(self, t, x):
        """
        The function interface used by scipy.ode

        Parameters
        ----------
        t : float
            The current time, t.
        x : np.array
            The 1D state vector.

        Returns
        -------
        xdot : np.array
            The 1D vector of state time-derivatives.

        """
        self.prob['time'] = t
        self.prob['time_phase'] = t - self.t_initial
        self._unpack_state_vec(x)
        self.prob.run_model()
        xdot = self._pack_state_rate_vec()
        return xdot