# PU-Ray
Official implementation of "PU-Ray: Point Cloud Upsampling via Ray Marching on Implicit Surface".

![](./supplementary/camel.gif)

## Installation
### Create virtual environment and install dependencies
```
conda create -n pu-ray python==3.8.17
conda activate pu-ray
pip install -r requirements.txt
pip install .
```
To check the installation:
```
pip freeze | grep pu-ray
```

