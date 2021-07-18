from setuptools import find_packages, setup

setup(
    name="spiking-neural-network",
    version="0.0.1",
    description="Practice for spiking neural network (SNN)",
    python_requires=">=3.7",
    install_requires=["numpy"],
    url="https://github.com/cosmoquester/spiking-neural-network.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
