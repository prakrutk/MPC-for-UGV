from setuptools import setup, find_packages

VERSION = '0.0.1'

INSTALL_REQUIRES = (
    [
        'numpy',
        'scipy',
        'jax',
        'pybullet',
        'matplotlib',
        'cvxpy',
        'cv2',
    ]
)

setup(
    name='MPC',
    version=VERSION,
    description='Model Predictive Control',
    author='Prakrut Kotecha',
    Install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires='>=3.8',
    zip_safe=False,
)