from setuptools import setup, find_packages

setup(
    name='EgoEMS',
    version='0.3',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    description='Custom PyTorch dataset class for EgoEMS dataset',
    author='Keshara Weerasinghe',
)
