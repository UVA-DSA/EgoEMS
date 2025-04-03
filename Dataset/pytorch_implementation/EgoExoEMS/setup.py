from setuptools import setup, find_packages

setup(
    name='EgoExoEMS',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    description='Custom PyTorch dataset class for EgoExoEMS dataset',
    author='Keshara Weerasinghe',
)
