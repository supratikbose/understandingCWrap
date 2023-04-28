# Understanding python wrapping of C Code, specially in relation to numpy and C conversion


## Installation
```
pip install git+https://github.com/supratikbose/understandingCWrap
```

## Usage
```
from cWrap import understandingCWrap
import SimpleITK as sitk

im1_in_np = sitk.GetArrayFromImage(sitk.ReadImage(im1_PATH))
im2_in_np = sitk.GetArrayFromImage(sitk.ReadImage(im2_PATH))

#In the return value   multiChannelRes_inOut_np is 3 channel numpy array

multiChannelRes_inOut_np = understandingCWrap(im1_in_np, im2_in_np)
```

## Prerequesities
Input image volumes must be numpy array having the same dimensions

## Development
Build:
```
python setup.py build_ext --inplace
```
## Test
Test 
```
python -m unittest 
```

