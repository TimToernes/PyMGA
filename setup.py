from setuptools import setup

setup(name='PyMGA',
      version='0.0.1',
      description='A Python library from Modeling to Generate Alternatives methods',
      url='#',
      author='Tim Pedersen',
      author_email='timtoernes@gmail.com',
      license='MIT',
      packages=['PyMGA'],
      install_requires=['numpy>=1.22',
                        'pypsa=0.21'
                        'matplotlib',
                        'gurobi=9.5',
                        'pandas',
                        'chaospy>=3.3',
                        'scipy',
                        'dask=2022.12',
                        'dask-jobqueue>=0.8'],
      zip_safe=False)
