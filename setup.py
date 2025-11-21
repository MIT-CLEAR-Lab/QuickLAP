from setuptools import find_packages, setup

setup(
    name="multimodal_intervention",
    packages=find_packages(include=["interact_drive", "experiments", "user_study"]),
    version="1.0",
    license="MIT",
)
