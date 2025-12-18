from setuptools import setup, find_packages

setup(
    name="IFT712_Project",
    version="0.1.0",
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    packages=find_packages("src"),
    author="Robin BECARD, Adrien SKRZYPCZAK, Cédric HAN",
    author_email="robinbecard@gmail.com, auteur1@example.com, auteur2@example.com",
    description="A project for IFT712, including machine learning components.",
)
