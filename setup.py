from setuptools import setup, find_packages
from typing import List

with open("README.md", 'r', encoding="utf-8") as f:
    long_description = f.read()

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


__VERSION__ = "0.0.0"

REPO_NAME = "KAGGLE_PS3E9"
AUTHOR_USER_NAME = "vikramviky123"
SRC_REPO = "concrete_strength"
AUTHOR_EMAIL = "vikram_viky2001@yahoo.com"


setup(
    name=SRC_REPO,
    version=__VERSION__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for mlflow",
    long_description=long_description,

    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements('requirements.txt')
)
