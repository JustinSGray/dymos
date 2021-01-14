import openmdao.api as om
from ...models.atmosphere import USatm1976Comp
from .k_comp import KComp
from .aero_forces_comp import AeroForcesComp
from .lift_coef_comp import LiftCoefComp
from dymos.models.eom import FlightPathEOM2D
from .stall_speed_comp import StallSpeedComp

class TakeoffODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'], 
                           promotes_outputs=['rho'])

        self.add_subsystem(name='k_comp',
                           subsys=KComp(num_nodes=nn),
                           promotes_inputs=['AR', 'span', 'e', 'h', 'h_w'],
                           promotes_outputs=['K'])

        self.add_subsystem(name='lift_coef_comp',
                           subsys=LiftCoefComp(num_nodes=nn),
                           promotes_inputs=['alpha', 'alpha_max', 'CL0', 'CL_max'],
                           promotes_outputs=['CL'])

        self.add_subsystem(name='aero_force_comp',
                           subsys=AeroForcesComp(num_nodes=nn),
                           promotes_inputs=['rho', 'v', 'S', 'CL', 'CD0', 'K'],
                           promotes_outputs=['q', 'L', 'D'])

        # Note: Typically a propulsion subsystem would go here, and provide thrust and mass
        # flow rate of the aircraft (for integrating mass).
        # In this simple demonstration, we're assuming the thrust of the aircraft is constant
        # and that the aircraft mass doesn't change (fuel burn during takeoff is negligible).

        self.add_subsystem(name='dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha', 'L', 'D', 'T'],
                           promotes_outputs=['h_dot', 'r_dot', 'v_dot', 'gam_dot'])

        # Add an ExecComp to compute weight here, since the input variable is mass
        self.add_subsystem('weight_comp',
                           subsys=om.ExecComp('W = 9.80665 * m', has_diag_partials=True,
                                              W={'units': 'N', 'shape': (nn,)},
                                              m={'units': 'kg', 'shape': (nn,)}),
                           promotes_inputs=['m'],
                           promotes_outputs=['W'])

        self.add_subsystem(name='stall_speed_comp',
                           subsys=StallSpeedComp(num_nodes=nn),
                           promotes_inputs=['v', 'W', 'rho', 'CL_max', 'S'],
                           promotes_outputs=['v_stall', 'v_over_v_stall'])

        self.set_input_defaults('CL_max', val=2.0)
        self.set_input_defaults('alpha_max', val=10.0, units='deg')


# import numpy as np
# from scipy.interpolate import interp1d

# import openmdao.api as om

# from .k_comp import KComp
# from .aero_forces_comp import AeroForcesComp
# from .lift_coef_comp import LiftCoefComp
# from dymos.models.eom import FlightPathEOM2D
# from .stall_speed_comp import StallSpeedComp



# import dymos as dm
# from dymos.models.atmosphere.atmos_1976 import USatm1976Data


# # CREATE an atmosphere interpolant
# english_to_metric_rho = om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
# english_to_metric_alt = om.unit_conversion('ft', 'm')[0]
# rho_interp = interp1d(np.array(USatm1976Data.alt*english_to_metric_alt, dtype=complex), 
#                       np.array(USatm1976Data.rho*english_to_metric_rho, dtype=complex), kind='linear')


# class TakeoffODE(om.ExplicitComponent):

#     def initialize(self):
#         self.options.declare('num_nodes', types=int)

#     def setup(self):
#         nn = self.options['num_nodes']


#         # parameters
#         self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
#         self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
#         self.add_input('span', val=35.7, desc='Wingspan', units='m')
#         self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
#         self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
#         self.add_input('CL_max', val=2.0, desc='maximum lift coefficient', units=None)
#         self.add_input('alpha_max', val=0.174533, desc='angle of attack at CL_max', units='rad') # 10 degrees
#         self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
#         self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)

#         # mass could be a state if we were tracking fuel burn, but we're not for this problem
#         # so it will a fixed parameter
#         # self.add_input('m', val=1000., desc='aircraft mass', units='kg')
#         self.add_input('m', shape=nn, desc='aircraft mass', units='kg')

#         # self.add_output('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
#         #self.add_output('L', val=np.ones(nn), desc='lift', units='N')
#         #self.add_output('D', val=np.ones(nn), desc='drag', units='N')

#         # controls 
#         self.add_input('T', val=np.zeros(nn), desc='thrust', units='N')
#         self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='rad')

#         # states 
#         self.add_input('gam', val=np.zeros(nn), desc='flight path angle', units='rad')
#         self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
#         self.add_input('h', shape=nn,  desc='altitude', units='m')

#         # state rates
#         self.add_output('v_dot', val=np.zeros(nn), desc='rate of change of velocity magnitude', units='m/s**2')
#         self.add_output('gam_dot', val=np.zeros(nn), desc='rate of change of flight path angle', units='rad/s')
#         self.add_output('h_dot', val=np.zeros(nn), desc='rate of change of altitude', units='m/s')
#         self.add_output('r_dot', val=np.zeros(nn), desc='rate of change of range', units='m/s')

#         # Extra outputs
#         self.add_output('v_over_v_stall', val=np.ones(nn), desc='stall speed ratio', units=None)


#         partials_method = 'fd'
#         self.declare_partials('*', '*', method=partials_method)
#         self.declare_coloring(wrt='*', method=partials_method, tol=1.0E-12, show_summary=False, show_sparsity=False)


#     def compute(self, inputs, outputs): 

#         h = inputs['h']
#         h_w = inputs['h_w']
#         span = inputs['span']
#         AR = inputs['AR']
#         e = inputs['e']
#         b = span / 2.0

#         # drag-due-to-lift factor to account for ground effect
#         K_nom = 1.0 / (np.pi * AR * e)
#         K = K_nom * 33 * ((h + h_w) / b)**1.5 / (1.0 + 33 * ((h + h_w) / b)**1.5)

#         ####################################
#         # get density from atmo interpolant 
#         ####################################
#         # handle complex-step gracefully from the interpolant
#         if np.iscomplexobj(h): 
#             rho = rho_interp(h)
#         else: 
#             rho = rho_interp(h).real
#         q = 0.5*rho*inputs['v']**2

#         ####################################
#         # Assume linear lift curve slope
#         ####################################
#         CL0 = inputs['CL0']
#         alpha = inputs['alpha']
#         alpha_max = inputs['alpha_max']
#         CL_max = inputs['CL_max']
#         CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)


#         ####################################
#         # Assume simple quadratic drag polar
#         #################################### 
#         v = inputs['v']
#         S = inputs['S']
#         CD0 = inputs['CD0']
#         q = 0.5*rho*inputs['v']**2
#         L = q * S * CL
#         D = q * S * (CD0 + K * CL ** 2)

#         ####################################
#         # Compute stall speed ratio
#         ####################################
#         GRAVITY = 9.80665
#         m = inputs['m']
        
#         W = m*GRAVITY
#         v_stall = np.sqrt(2 * W / rho / S / CL_max)
#         outputs['v_over_v_stall'] = v / v_stall

#         ####################################
#         # 2D aircraft EOM
#         ####################################
#         T = inputs['T']
#         gam = inputs['gam']
#         alpha = inputs['alpha']

#         calpha = np.cos(alpha)
#         salpha = np.sin(alpha)
#         cgam = np.cos(gam)
#         sgam = np.sin(gam)
#         mv = m * v


#         outputs['v_dot'] = (T * calpha - D) / m - GRAVITY * sgam
#         outputs['gam_dot'] = (T * salpha + L) / mv - (GRAVITY / v) * cgam
#         outputs['h_dot'] = v * sgam
#         outputs['r_dot'] = v * cgam