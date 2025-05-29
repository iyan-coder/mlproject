from setuptools import find_packages, setup
from typing import List

# Constant for editable install reference
HYPEN_E_DOT = "-e ."

# ------------------------------------------------------
# Function to read dependencies from requirements.txt
# ------------------------------------------------------
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements.txt file and returns a clean list of required packages.
    It also removes the editable install line (-e .) if present.
    '''
    requirements = []
    with open(file_path) as file_obj:
        # Read all lines instead of just the first one (fix bug)
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        # Remove '-e .' if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# ------------------------------------------------------
# Setup configuration for the ML project
# ------------------------------------------------------
setup(
    name="mlproject",
    version="0.0.1",
    author="iyanuoluwa",
    author_email="iyanuolouwaadebayo04@gmail.com",
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=get_requirements("requirements.txt")  # List of dependencies
)
