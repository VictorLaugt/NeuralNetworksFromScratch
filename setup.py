from distutils.core import setup
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIREMENTS_FILE = HERE.joinpath('requirements.txt')

with REQUIREMENTS_FILE.open(mode='r') as requirements:
    install_requires = requirements.read().splitlines()

setup(
    name="minimalist_neural_networks",
    version="1.1",
    description=(
        "A minimalist implementation of deep neural networks which requires few "
        "dependencies."
    ),
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.11'
    ],
    keywords='deep machine learning minimalist neural network',

    author="Victor Laügt",
    author_email='victorlaugtdev@gmail.com',
    url="https://github.com/VictorLaugt/NeuralNetworksFromScratch",
    license='GPLv3',

    packages=[
        'neural',
        'neural.functions',
        'neural.structures'
    ],
    install_requires=install_requires,

    include_package_data=True,
)
