from setuptools import setup

setup(
    name='MAPPER_implementation',
    version='0.0.1',
    install_requires=[
        # Simulation evironments:
        'gym',
        'mlagents',  # Connection to unity
        # Tensorboard
        # 'tensorboardX',  # For pytorch
        # 'tensorboard',  # For tensorflow
        # Pytorch
        'torch==1.9.1+cu111',
        'torchvision==0.10.1+cu111',
        'torchaudio==0.9.1',
        # Pytorch for GNN
        'torch-scatter',
        'torch-sparse',
        'torch-cluster',
        'torch-spline-conv',
        'torch-geometric',
        # Image processing libraries
        'opencv-python',
        'scikit-image',
        'colorhash',
        'numpy',
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
        'https://data.pyg.org/whl/torch-1.9.0+cu102.html'
    ]
)
