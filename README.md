# MattingZoom
A PyTorch-based library for trimap-free matting.


 
## Supported Models:
- [x] SHM
- [x] U2Net
- [x] MODNet
- [x] GFM

## The requirement for extending custom method.
- model file: 
  - Put the model file in the *models* directory.
  - The name of model file should follow **[custom]_net**.
  - The class name should follow **[custom]_Net**.
  - The parameter of __init__ of **[custom]_Net** must is **args**.
  - The last return of **forward** must is the **matte** ~ [0,1].
