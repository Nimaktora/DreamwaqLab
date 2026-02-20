from setuptools import find_packages, setup

setup(
    name="DreamwaqLab",
    version="0.1.0",
    author="JaeyeolKim",
    author_email="9191kjy@gmail.com",
    description="",
    packages=find_packages(include=["exts.dreamwaq", "exts.dreamwaq.*"]),
    install_requires=[
        "torch>=2.5.1",
        "gymnasium",
        "numpy",
        "torchvision",
        "omegaconf",
        "hydra-core",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
    ],
)
