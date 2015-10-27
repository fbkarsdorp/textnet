import setuptools

from distutils.core import setup 

setup(
    name='textnet',
    version='0.0.1',
    packages=['textnet'],
    url='http://fbkarsdorp.github.io/textnet',
    author='Folgert Karsdorp',
    author_email='fbkarsdorp AT fastmail DOT nl',
    install_requires=['numpy', 'scipy', 'pandas', 'networkx', 'scikit-learn', 'igraph', 'seaborn', 'pyprind']
)


