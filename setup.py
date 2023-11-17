from setuptools import setup, find_packages

VERSION = '0.1.0'

INSTALL_REQUIRES = (
    [
        'numpy',
        'scipy',
        'jax',
        'pybullet',
        'matplotlib',
    ]
)

setup(
    name='MPC',
    version=VERSION,
    description='Model Predictive Control',
    author='Prakrut Kotecha',
    Install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    zip_safe=False,
)
