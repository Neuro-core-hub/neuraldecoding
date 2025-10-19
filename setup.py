import os
from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as file:
    requirements = file.read().splitlines()

setup(
    name='neuraldecoding',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='',
    author_email='',
    description='A package for neural decoding applications.',
    url='https://github.com/Neuro-core-hub/neuraldecoding/',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)