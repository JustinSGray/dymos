from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp 
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.optimizer_based.components.collocation_balance_comp import CollocationBalanceComp
from dymos.phases.grid_data import GridData


class TestCollocationBalanceIndex(unittest.TestCase): 

    def make_prob(self, transcription, n_segs, order, compressed): 

        p = Problem(model=Group())

        gd = GridData(num_segments=n_segs, segment_ends=np.arange(n_segs+1),
                      transcription=transcription, transcription_order=order, compressed=compressed)

        state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial':True, 'fix_final':False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial':False, 'fix_final':True}}

        defect_comp = p.model.add_subsystem('defect_comp',
                                            subsys=CollocationBalanceComp(grid_data=gd,
                                                                          state_options=state_options))

        p.setup()
        p.final_setup()

        return p


    def test_3_lgl(self): 

        p = self.make_prob(transcription='gauss-lobatto', n_segs=3, order=3, compressed=False)
        defect_comp = p.model.defect_comp
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1, 3, 5]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 2, 4]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0,1,3]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([2,4,5]))


    def test_5_lgl(self): 

        p = self.make_prob(transcription='gauss-lobatto', n_segs=2, order=5, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1, 2, 4, 5]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 3]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0, 1, 2, 4]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([3, 5]))


    def test_3_lgr(self): 

        p = self.make_prob(transcription='radau-ps', n_segs=3, order=3, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1,2,3,5,6,7,9,10,11]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 4, 8]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0,1,2,3,5,6,7,9,10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([4, 8, 11]))


    def test_5_lgr(self): 

        p = self.make_prob(transcription='radau-ps', n_segs=2, order=5, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1,2,3,4,5,7,8,9,10,11]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 6]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0,1,2,3,4,5,7,8,9,10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([6, 11]))

    def test_5_lgr_compressed(self): 

        p = self.make_prob(transcription='radau-ps', n_segs=2, order=5, compressed=True)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1,2,3,4,5,6,7,8,9,10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0,]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0,1,2,3,4,5,6,7,8,9]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([10,]))


    def test_5_lgl_compressed(self): 

        p = self.make_prob(transcription='gauss-lobatto', n_segs=2, order=5, compressed=True)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1,2,3,4]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0,]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0,1,2,3]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([4,]))

class TestCollocationBalanceApplyNL(unittest.TestCase):

    def setUp(self):
        transcription = 'gauss-lobatto'

        gd = GridData(
            num_segments=4, segment_ends=np.array([0., 2., 4., 5., 12.]),
            transcription=transcription, transcription_order=3)

        self.p = Problem(model=Group())

        state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial':True, 'fix_final':False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial':True, 'fix_final':False}}

        indep_comp = IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

        indep_comp.add_output(
            'dt_dstau',
            val=np.ones((gd.subset_num_nodes['col']))
        )

        indep_comp.add_output(
            'f_approx:x',
            val=np.zeros((gd.subset_num_nodes['col'], 1)), units='m')
        indep_comp.add_output(
            'f_computed:x',
            val=np.zeros((gd.subset_num_nodes['col'], 1)), units='m')

        indep_comp.add_output(
            'f_approx:v',
            val=np.zeros((gd.subset_num_nodes['col'], 3, 2)), units='m/s')
        indep_comp.add_output(
            'f_computed:v',
            val=np.zeros((gd.subset_num_nodes['col'], 3, 2)), units='m/s')

        self.p.model.add_subsystem('defect_comp',
                                   subsys=CollocationBalanceComp(grid_data=gd,
                                                                 state_options=state_options))

        self.p.model.connect('f_approx:x', 'defect_comp.f_approx:x')
        self.p.model.connect('f_approx:v', 'defect_comp.f_approx:v')
        self.p.model.connect('f_computed:x', 'defect_comp.f_computed:x')
        self.p.model.connect('f_computed:v', 'defect_comp.f_computed:v')
        self.p.model.connect('dt_dstau', 'defect_comp.dt_dstau')

        self.p.setup(force_alloc_complex=True)

        self.p['dt_dstau'] = np.random.random(gd.subset_num_nodes['col'])

        self.p['f_approx:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        self.p['f_approx:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        self.p['f_computed:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        self.p['f_computed:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        self.p.run_model()
        self.p.model.run_apply_nonlinear() # need to make sure residuals are computed

    def test_results(self):
        dt_dstau = self.p['dt_dstau']

        assert_almost_equal(self.p.model._residuals._views['defect_comp.x'],
                            dt_dstau[:, np.newaxis] * 
                            (self.p['f_approx:x']-self.p['f_computed:x']))

        assert_almost_equal(self.p.model._residuals._views['defect_comp.v'],
                            dt_dstau[:, np.newaxis, np.newaxis] *
                            (self.p['f_approx:v']-self.p['f_computed:v']))

       

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd')
        # assert_check_partials(cpd)

        print((self.p['f_approx:v']-self.p['f_computed:v']).ravel())


if __name__ == '__main__':
    unittest.main()
