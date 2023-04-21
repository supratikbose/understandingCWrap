from libcpp cimport bool

import SimpleITK as sitk
import numpy as np

cimport numpy as np

cdef extern from "libs/understandingCWrap.h":
    void understandingCWrap_API01(float *im1_in, float *im2_in, float *multiChannelRes_inOut, int dim3, int dim2, int dim1)

def understandingCWrap_API01_cpp(np.ndarray[np.float32_t, ndim=1] im1_in,
                                 np.ndarray[np.float32_t, ndim=1] im2_in,
                                 np.ndarray[np.float32_t, ndim=1] multiChannelRes_inOut,
                                 shape):
    return understandingCWrap_API01(&im1_in[0], &im2_in[0], &multiChannelRes_inOut[0], shape[2], shape[1], shape[0])

def understandingCWrap(im1_in, im2_in):
    origin_type = im1_in.dtype
    shape = im1_in.shape
    multiChannelRes_inOut_1channel=np.zeros(im1_in.shape)
    multiChannelRes_inOut=np.repeat(multiChannelRes_inOut_1channel[np.newaxis,...], 3, axis=0)
    multiChannelRes_inOut_shape=multiChannelRes_inOut.shape 

    im1_in = im1_in.flatten().astype(np.float32)
    im2_in = im2_in.flatten().astype(np.float32)
    multiChannelRes_inOut=multiChannelRes_inOut.flatten().astype(np.float32)    

    understandingCWrap_API01_cpp(im1_in, im2_in, multiChannelRes_inOut, shape)

    multiChannelRes_inOut = np.reshape(multiChannelRes_inOut, multiChannelRes_inOut_shape).astype(origin_type)

    return multiChannelRes_inOut

    

# cdef extern from "libs/deedsBCV0.h":
#     void deeds(float *im1, float *im1b, float *warped1, float *flow, int m, int n, int o, float alpha, int levels, bool verbose)


# def deeds_cpp(np.ndarray[np.float32_t, ndim=1] fixed,
#               np.ndarray[np.float32_t, ndim=1] moving,
#               np.ndarray[np.float32_t, ndim=1] moved,
#               np.ndarray[np.float32_t, ndim=1] flow_3channel,
#               shape, alpha, level, verbose):
#     return deeds(&moving[0], &fixed[0], &moved[0], &flow_3channel[0],
#                   shape[2], shape[1], shape[0],
#                   alpha, level, verbose)


# def registration(fixed_vol_np, moving_vol_np, alpha=1.6, levels=5, verbose=True):
#     fixed_np = fixed_vol_np.copy() #to_numpy(fixed)
#     moving_np = moving_vol_np.copy() #to_numpy(moving)

#     origin_type = moving_np.dtype

#     shape = moving_np.shape
    
#     flow_1channel_np=np.zeros(moving_np.shape)
#     flow_3channel_np=np.repeat(flow_1channel_np[np.newaxis,:,:,:], 3, axis=0)
#     flow_3channel_shape=flow_3channel_np.shape    
    
#     fixed_np = fixed_np.flatten().astype(np.float32)
#     moving_np = moving_np.flatten().astype(np.float32)
#     moved_np = np.zeros(moving_np.shape).flatten().astype(np.float32)
#     flow_3channel_np=flow_3channel_np.flatten().astype(np.float32)

#     deeds_cpp(fixed_np, moving_np, moved_np, flow_3channel_np, shape, alpha, levels, verbose)

#     moved_np = np.reshape(moved_np, shape).astype(origin_type)
#     flow_3channel_np = np.reshape(flow_3channel_np, flow_3channel_shape).astype(origin_type)

#     moved_vol_np = moved_np.copy() #to_sitk(moved_np, ref_img=fixed)
#     return moved_vol_np, flow_3channel_np


# def to_numpy(img):
#     result = sitk.GetArrayFromImage(img)

#     return result


# def to_sitk(img, ref_img=None):
#     img = sitk.GetImageFromArray(img)

#     if ref_img:
#         img.CopyInformation(ref_img)

#     return img
