from setuptools import setup

setup(
    name='GausSN',
    version='0.1.0',    
    description='A package for fitting gravitationally lensed supernova light curves and other astronomical data with Gaussian Processes.',
    url='https://github.com/erinhay/GausSN',
    author='Erin Hayes',
    author_email='eeh55@cam.ac.uk',
    packages=['GausSN'],
    install_requires=['numpy', 'jax', 'matplotlib'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)
