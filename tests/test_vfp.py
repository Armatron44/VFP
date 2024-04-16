from vfp import vfp
import numpy as np
from numpy.testing import assert_allclose
import scipy

class test_settings():
    def __init__(self):
        # set up some quick standard test parameters.
        # 4 layers + use some non-integer values.
        self.lot = [0, 19.7, 50, 30]
        self.lor = [3.1, 5, 7.3, 6]
        self.nSLDs = [0, 6, 4, 3.47, 2.07]

    def test_calc_zeds(self):
        """
        Test calc_zeds of vfp.
        """ 
    
        des_start = -17.5 # -5 - (4 * 3.1) = -17.4 before using floor.
        des_end = 134 # 5 + (0 + 19.7 + 50 + 30) + 7.3 * 4 = 133.9 before using ceil.
        # number of points = ((17.5 + 134) / 0.5) + 1 = 304
        desired = np.linspace(des_start, des_end, 304)

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        # output from calc_zeds is passed over to z_and_SLD_scatter.
        actual = test_object.z_and_SLD_scatter(reduced=False)[0]
        
        assert_allclose(actual, desired)

    def test_SLD_offset(self):
        """
        Test SLD_offset of vfp.
        """
        desired_front_offset = -17.5 # -5 - (4 * 3.1) = -17.4 before using floor.

        # np.ceil((0 + 19.7 + 50 + 30) + 7.3 * 4) + 5 + dz[-1]( = 0.5) = 134.5.
        # Then transform to be at the front. 
        # 99.7 is now z = 0, thus the starting point should be the -ve distance between end point and total thickness.
        # Distance between end point (134.5) and total thickness (99.7) = 34.8
        desired_back_offset = -34.8 

        test_object_front = vfp(nucSLDs=self.nSLDs,
                                thicknesses=self.lot,
                                roughnesses=self.lor)
        
        test_object_back = vfp(nucSLDs=self.nSLDs,
                            thicknesses=self.lot,
                            roughnesses=self.lor,
                            orientation='back')
        
        actual_front_offset = test_object_front.SLD_offset()
        assert(actual_front_offset == desired_front_offset)

        actual_back_offset = test_object_back.SLD_offset()
        assert(actual_back_offset == desired_back_offset)

    def test_calc_vfp(self):   
        """
        Test calc_vfp of vfp.
        Will test all possible permutations of conformal for 4 interfaces.
        """

        start = -17.5 # -5 - (4 * 3.1) = -17.4 before using floor.
        end = 134 # 5 + (0 + 19.7 + 50 + 30) + 7.3 * 4 = 133.9 before using ceil.

        z = np.linspace(start, end, 304)

        # conformal = [0, 0, 0, 0]
        
        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)
        first_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * (1-scipy.stats.norm.cdf(z, loc=19.7, scale=5))
        second_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * scipy.stats.norm.cdf(z, loc=19.7, scale=5) * (1-scipy.stats.norm.cdf(z, loc=50, scale=7.3))
        third_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * scipy.stats.norm.cdf(z, loc=19.7, scale=5) * scipy.stats.norm.cdf(z, loc=50, scale=7.3) * (1-scipy.stats.norm.cdf(z, loc=30, scale=6))
        backing = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * scipy.stats.norm.cdf(z, loc=19.7, scale=5) * scipy.stats.norm.cdf(z, loc=50, scale=7.3) * scipy.stats.norm.cdf(z, loc=30, scale=6)

        # now stack each layer vf profile row wise.
        non_conf_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        non_conf_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                tuple(self.lot), 
                                                tuple(self.zeds), 
                                                (0, 0, 0, 0))
        
        assert_allclose(non_conf_actual_vfp, non_conf_desired_vfp)

        # conformal = [0, 1, 0, 0]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)

        # calculate the offset for the remaining layers
        F0st1 = scipy.stats.norm.cdf(z, loc=0+19.7, scale=3.1)
        first_lay = (1-F0st1) - first_lay
        second_lay = F0st1 * (1-scipy.stats.norm.cdf(z, loc=50, scale=7.3))
        third_lay = F0st1 * scipy.stats.norm.cdf(z, loc=50, scale=7.3) * (1-scipy.stats.norm.cdf(z, loc=30, scale=6))
        backing = 1-(fronting, first_lay, second_lay, third_lay)

        # now stack each layer vf profile row wise.
        conf_0100_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0100_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 1, 0, 0))
        
        assert_allclose(conf_0100_actual_vfp, conf_0100_desired_vfp)

        # conformal = [0, 0, 1, 0]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)
        first_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * (1-scipy.stats.norm.cdf(z, loc=19.7, scale=5))

        # calculate the offsets for the remaining layers
        F0st2 = scipy.stats.norm.cdf(z, loc=0+50, scale=3.1)
        F1st2 = scipy.stats.norm.cdf(z, loc=19.7+50, scale=5)

        # now calc vf profiles for every layer after conformal interface.
        second_lay = (1-(F0st2*F1st2)) - (fronting + first_lay)
        third_lay = F0st2*F1st2 * (1-scipy.stats.norm.cdf(z, loc=30, scale=6))
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0010_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0010_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 0, 1, 0))
        
        assert_allclose(conf_0010_actual_vfp, conf_0010_desired_vfp)

        # conformal = [0, 0, 0, 1]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)
        first_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * (1-scipy.stats.norm.cdf(z, loc=19.7, scale=5))
        second_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * scipy.stats.norm.cdf(z, loc=19.7, scale=5) * (1-scipy.stats.norm.cdf(z, loc=50, scale=7.3))

        # calculate the offsets for the remaining layers
        F0st3 = scipy.stats.norm.cdf(z, loc=0+30, scale=3.1)
        F1st3 = scipy.stats.norm.cdf(z, loc=19.7+30, scale=5)
        F2st3 = scipy.stats.norm.cdf(z, loc=19.7+50+30, scale=7.3)

        # now calc vf profiles for every layer after conformal interface.
        third_lay = (1-(F0st3*F1st3*F2st3)) - (fronting + first_lay + second_lay)
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0001_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0001_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 0, 0, 1))
        
        assert_allclose(conf_0001_actual_vfp, conf_0001_desired_vfp)

        # conformal = [0, 1, 1, 0]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)

        # calculate the offsets for the conformal interfaces
        F0st1 = scipy.stats.norm.cdf(z, loc=0+19.7, scale=3.1)
        F0st1t2 = scipy.stats.norm.cdf(z, loc=0+19.7+50, scale=3.1)

        # now calc vf profiles for every layer after first conformal interface.
        first_lay = (1-F0st1) - fronting
        second_lay = (1-F0st1t2) - (fronting + first_lay)
        third_lay = F0st1t2 * (1-scipy.stats.norm.cdf(z, loc=30, scale=6))
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0110_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0110_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 1, 1, 0))
        
        assert_allclose(conf_0110_actual_vfp, conf_0110_desired_vfp)

        # conformal = [0, 0, 1, 1]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)
        first_lay = scipy.stats.norm.cdf(z, loc=0, scale=3.1) * 1-scipy.stats.norm.cdf(z, loc=19.7, scale=5)

        # calculate the offsets for the conformal interfaces
        F0st2 = scipy.stats.norm.cdf(z, loc=0+50, scale=3.1)
        F1st2 = scipy.stats.norm.cdf(z, loc=0+19.7+50, scale=5)
        F0st2t3 = scipy.stats.norm.cdf(z, loc=0+50+30, scale=3.1)
        F1st2t3 = scipy.stats.norm.cdf(z, loc=0+19.7+50+30, scale=5)

        # now calc vf profiles for every layer after first conformal interface.
        second_lay = (1-(F0st2*F1st2)) - (fronting + first_lay)
        third_lay = (1-(F0st2t3*F1st2t3)) - (fronting + first_lay + second_lay)
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0011_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0011_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 0, 1, 1))
        
        assert_allclose(conf_0011_actual_vfp, conf_0011_desired_vfp)

        # conformal = [0, 1, 0, 1]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)

        # calculate the offsets for the conformal interfaces
        F0st1 = scipy.stats.norm.cdf(z, loc=0+19.7, scale=3.1)
        F2st3 = scipy.stats.norm.cdf(z, loc=50+30, scale=7.3)

        # now calc vf profiles for every layer after first conformal interface.
        first_lay = (1-F0st1) - fronting
        second_lay = F0st1 * (1-scipy.stats.norm.cdf(z, loc=50, scale=7.3))
        third_lay = (1-F0st1*F2st3) - (fronting + first_lay + second_lay)
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0101_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0101_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 1, 0, 1))
        
        assert_allclose(conf_0101_actual_vfp, conf_0101_desired_vfp)

        # conformal = [0, 1, 1, 1]

        # calculate vf profiles for each layer.
        fronting = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)

        # calculate the offsets for the conformal interfaces
        F0st1 = scipy.stats.norm.cdf(z, loc=0+19.7, scale=3.1)
        F0st1st2 = scipy.stats.norm.cdf(z, loc=0+19.7+50, scale=3.1)
        F0st1st2st3 = scipy.stats.norm.cdf(z, loc=0+19.7+50+30, scale=3.1)

        # now calc vf profiles for every layer after first conformal interface.
        first_lay = (1-F0st1) - fronting
        second_lay = (1-F0st1st2) - (fronting + first_lay)
        third_lay = (1-F0st1st2st3) - (fronting + first_lay + second_lay)
        backing = 1-(fronting + first_lay + second_lay + third_lay)

        # now stack each layer vf profile row wise.
        conf_0111_desired_vfp = np.vstack(fronting, first_lay, second_lay, third_lay, backing) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        conf_0111_actual_vfp = test_object.calc_vfp(tuple(self.lor), 
                                                    tuple(self.lot), 
                                                    tuple(self.zeds), 
                                                    (0, 1, 1, 1))
        
        assert_allclose(conf_0111_actual_vfp, conf_0111_desired_vfp)

    def test_one_minus_cdf(self):   
        """
        Test one_minus_cdf of vfp.
        """

        start = -17.5 # -5 - (4 * 3.1) = -17.4 before using floor.
        end = 134 # 5 + (0 + 19.7 + 50 + 30) + 7.3 * 4 = 133.9 before using ceil.

        z = np.linspace(start, end, 304)

        desired_one_minus_F0 = 1-scipy.stats.norm.cdf(z, loc=0, scale=3.1)
        desired_one_minus_F1 = 1-scipy.stats.norm.cdf(z, loc=19.7, scale=5)
        desired_one_minus_F2 = 1-scipy.stats.norm.cdf(z, loc=50, scale=7.3)
        desired_one_minus_F3 = 1-scipy.stats.norm.cdf(z, loc=30, scale=6) 

        test_object = vfp(nucSLDs=self.nSLDs,
                        thicknesses=self.lot,
                        roughnesses=self.lor)
        
        zeds = test_object.z_and_SLD_scatter(reduced=False)[0]
        thick = np.cumsum(self.lot)

        actual_one_minus_F0 = test_object.one_minus_cdf(interf_choice=0, x=zeds, cumthick=thick, rough=self.lor)
        actual_one_minus_F1 = test_object.one_minus_cdf(interf_choice=1, x=zeds, cumthick=thick, rough=self.lor)
        actual_one_minus_F2 = test_object.one_minus_cdf(interf_choice=2, x=zeds, cumthick=thick, rough=self.lor)
        actual_one_minus_F3 = test_object.one_minus_cdf(interf_choice=3, x=zeds, cumthick=thick, rough=self.lor)

        assert_allclose(actual_one_minus_F0, desired_one_minus_F0)
        assert_allclose(actual_one_minus_F1, desired_one_minus_F1)
        assert_allclose(actual_one_minus_F2, desired_one_minus_F2)
        assert_allclose(actual_one_minus_F3, desired_one_minus_F3)

    # TODO: finish tests for all other non-simple functions.

    # def test_calc_dzs(self):
    #     """
    #     Test calc_dzs of vfp.
    #     """
        
    #     test_object = vfp(nucSLDs=self.nSLDs,
    #                     thicknesses=self.lot,
    #                     roughnesses=self.lor)

    #     zstart, zend, points, _ = test_object.calc_zeds(tuple(self.lor), tuple(self.lot), 0.5)

    #     actual_dz = test_object.calc_dzs(zstart, zend, points, self.indices)

    #     assert_allclose(actual_dz, desired_one_minus_F0)




