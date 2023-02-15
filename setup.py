from setuptools import find_packages,setup
from typing import List 

REQUIREMENT_FILE="requirements.txt"
HYPHEN_E='-e .'


def get_requirements():
    with open(REQUIREMENT_FILE) as file:
        requirement_list=file.read_lines()
        requirement_list=[requirement_name.replace('\n',' ') for requirement_name in requirement_list]
        if HYPHEN_E in requirement_list:
            requirement_list.remove(HYPHEN_E)
        return requirement_list


setup(
    name="fraud_transaction_detector",
    version='0.0.1',
    author="mahammadrafi",
    author_email="mahammadrafishaik222@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)