from __future__ import print_function, division, absolute_import

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase, Trajectory
from dymos.examples.ballistic_rocket.ballistic_rocket_ode import BallisticRocketUnguidedODE, \
    BallisticRocketGuidedODE

SHOW_PLOTS = True


def ballistic_rocket_max_range(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                               run_driver=True, top_level_jacobian='csc', compressed=True,
                               sim_record='ballistic_rocket_sim.db', optimizer='SLSQP',
                               dynamic_simul_derivs=True):
    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
    else:
        p.driver = ScipyOptimizeDriver()

    p.driver.options['dynamic_simul_derivs'] = dynamic_simul_derivs

    #
    # The Trajectory Group
    #
    traj = Trajectory()

    #
    # Phase 0: Vertical Boost
    #

    boost_phase = Phase(transcription,
                  ode_class=BallisticRocketUnguidedODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('boost', boost_phase)

    boost_phase.set_time_options(fix_initial=True, duration_bounds=(0, 10))

    boost_phase.set_state_options('x', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('y', fix_initial=True, fix_final=True)
    boost_phase.set_state_options('vx', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('vy', fix_initial=True, fix_final=False)
    boost_phase.set_state_options('mprop', fix_initial=True, fix_final=False)

    boost_phase.add_design_parameter('thrust', units='N', opt=False, val=2000.0)
    boost_phase.add_design_parameter('theta', units='deg', opt=False, val=90.0)
    boost_phase.add_design_parameter('mstruct', units='kg', opt=False, val=100.0)
    boost_phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)
    boost_phase.add_design_parameter('Isp', units='s', opt=False, val=300.0)

    # boost_phase.add_objective('time_phase', loc='final', scaler=10)
    
    traj.add_phase('boost', boost_phase)

    #
    # Phase 1: Pitchover 
    #

    pitch_over_phase = Phase(transcription,
                             ode_class=BallisticRocketGuidedODE,
                             num_segments=num_segments,
                             transcription_order=transcription_order,
                             compressed=compressed)

    pitch_over_phase.set_time_options(fix_initial=False, duration_bounds=(0, 1000))

    pitch_over_phase.set_state_options('x', fix_initial=False, fix_final=False)
    pitch_over_phase.set_state_options('y', fix_initial=False, fix_final=False)
    pitch_over_phase.set_state_options('vx', fix_initial=False, fix_final=False)
    pitch_over_phase.set_state_options('vy', fix_initial=False, fix_final=False)
    pitch_over_phase.set_state_options('mprop', fix_initial=False, fix_final=True)

    pitch_over_phase.add_design_parameter('thrust', units='N', opt=False, val=2000.0)
    pitch_over_phase.add_design_parameter('mstruct', units='kg', opt=False, val=100.0)
    pitch_over_phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)
    pitch_over_phase.add_design_parameter('Isp', units='s', opt=False, val=300.0)
    pitch_over_phase.add_design_parameter('theta_0', units='deg', opt=False, val=90.0)
    pitch_over_phase.add_design_parameter('theta_f', units='deg', opt=False, val=45.0, lower=0, upper=89)

    pitch_over_phase.add_objective('time_phase', loc='final', scaler=10)
    
    traj.add_phase('pitch_over', pitch_over_phase)

    traj.link_phases(phases=['boost', 'pitch_over'], vars=['time', 'x', 'y', 'vx', 'vy', 'mprop'])

    p.model = Group()
    p.model.add_subsystem('traj', traj)

    #
    # Setup and set values
    #
    # p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True)
    p.final_setup()
    p['traj.boost.t_initial'] = 0.0
    p['traj.boost.t_duration'] = 2.0

    p['traj.boost.states:x'] = boost_phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj.boost.states:y'] = boost_phase.interpolate(ys=[0, 100], nodes='state_input')
    p['traj.boost.states:vx'] = boost_phase.interpolate(ys=[0, 0], nodes='state_input')
    p['traj.boost.states:vy'] = boost_phase.interpolate(ys=[0, 100], nodes='state_input')
    p['traj.boost.states:mprop'] = boost_phase.interpolate(ys=[20, 0], nodes='state_input')

    p['traj.boost.design_parameters:g'] = 9.80665
    p['traj.boost.design_parameters:theta'] = 90.0
    p['traj.boost.design_parameters:mstruct'] = 100

    p['traj.pitch_over.t_initial'] = 0.0
    p['traj.pitch_over.t_duration'] = 2.0

    p['traj.pitch_over.states:x'] = pitch_over_phase.interpolate(ys=[1, 10], nodes='state_input')
    p['traj.pitch_over.states:y'] = pitch_over_phase.interpolate(ys=[20, 100], nodes='state_input')
    p['traj.pitch_over.states:vx'] = pitch_over_phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj.pitch_over.states:vy'] = pitch_over_phase.interpolate(ys=[50, 100], nodes='state_input')
    p['traj.pitch_over.states:mprop'] = pitch_over_phase.interpolate(ys=[10, 1], nodes='state_input')

    p['traj.pitch_over.design_parameters:g'] = 9.80665
    p['traj.pitch_over.design_parameters:theta_0'] = 90.0
    p['traj.pitch_over.design_parameters:theta_f'] = 45.0
    p['traj.pitch_over.design_parameters:mstruct'] = 100

    p.run_driver()

    exp_out = traj.simulate()

    # Plot results
    if SHOW_PLOTS:
        # exp_out = boost_phase.simulate(times=50, record_file=sim_record)

        fig, axes = plt.subplots(ncols=2)
        fig.suptitle('Ballistic Rocket Solution')

        # plot the boost phase
        boost_phase = p.model.traj.phases.boost
        x_imp = boost_phase.get_values('x', nodes='all')
        y_imp = boost_phase.get_values('y', nodes='all')
        t_imp = boost_phase.get_values('time', nodes='all')
        
        x_exp = exp_out.get_values('x', phases='boost', flat=True)
        y_exp = exp_out.get_values('y', phases='boost', flat=True)
        t_exp = exp_out.get_values('time', phases='boost', flat=True)

        ax = axes[0]
        ax.plot(t_imp, y_imp, 'ro', label='i: boost')
        ax.plot(t_exp, y_exp, 'r-', label='sim: boost')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('y (m)')
        ax.grid(True)

        ax = axes[1]
        ax.plot(x_imp, y_imp, 'ro', label='i: boost')
        ax.plot(x_exp, y_exp, 'r-', label='sim: boost')
        
        # plot the pitch_over phase
        pitch_over = p.model.traj.phases.pitch_over
        x_imp = pitch_over.get_values('x', nodes='all')
        y_imp = pitch_over.get_values('y', nodes='all')
        t_imp = pitch_over.get_values('time', nodes='all')

        x_exp = exp_out.get_values('x', phases='pitch_over', flat=True)
        y_exp = exp_out.get_values('y', phases='pitch_over', flat=True)
        t_exp = exp_out.get_values('time', phases='pitch_over', flat=True)

        ax = axes[0]
        ax.plot(t_imp, y_imp, 'bo', label='i: pitch_over')
        ax.plot(t_exp, y_exp, 'b-', label='sim: pitch_over')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('y (m)')

        ax = axes[1]
        ax.plot(x_imp, y_imp, 'bo', label='i: pitch_over')
        ax.plot(x_exp, y_exp, 'b-', label='sim: pitch_over')
        ax.legend(loc='upper right')

     

        plt.show()

    return p


if __name__ == '__main__':
    ballistic_rocket_max_range(transcription='radau-ps', num_segments=10, run_driver=True,
                               transcription_order=3, compressed=True,
                               optimizer='SNOPT')
