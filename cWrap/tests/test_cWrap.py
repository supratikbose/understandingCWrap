import unittest
import numpy as np

from ..understandingCWrap import understandingCWrap


class TestStringMethods(unittest.TestCase):

    def test_understandingCWrap_API01(self):
        im1_in_np = np.random.randint(low=1, high=5, size=(2,3,3))
        im2_in_np = np.random.randint(low=6, high=10, size=(2,3,3))
        res = understandingCWrap(im1_in_np, im2_in_np)
        slice=1
        print(f'slice: {slice}')
        print(f'<<im1_in_np_sl_{slice}>>')
        print(f'{im1_in_np[:,:,slice]}')
        print(f'<<im2_in_np_sl_{slice}>>')
        print(f'{im2_in_np[:, :, slice]}')
        print(f'res_ch0_sl_{slice} to match im1_in_np_sl_{slice}')
        print(f'{res[0,:,:,slice]}')
        print(f'res_ch1_sl_{slice} to match im2_in_np_sl_{slice}')
        print(f'{res[1, :, :, slice]}')
        print(f'res_ch2_sl_{slice}  to be 2*I1-I2')
        print(f'{res[2, :, :, slice]}')
        
