from setuptools import setup, find_packages
setup(
    name='maxent',
    version='0.1',
    url='https://github.com/ezhenrik/maxent',
    author='Erik Henriksson',
    author_email='erik.ilmari.henriksson@gmail.com',
    description='A simple MaxEnt modeling package',
    packages=['maxent'],
    install_requires=['numpy', 'terminaltables', 'matplotlib', 'termplotlib']
)