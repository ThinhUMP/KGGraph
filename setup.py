from setuptools import setup, find_packages

setup(
    name='KGGraph',
    version='0.1.0',
    description='Knowledge Guided Graph for molecular representation',
    author='Van-Thinh-To',
    author_email='tvthinh.d19@ump.edu.vn',
    url='https://github.com/ThinhUMP/KGGraph',
    packages=find_packages(),
    install_requires=[
        'rdkit==2023.9.5',
        'networkx==3.2.1',
        'joblib==1.3.2'
    ],
    python_requires='==3.11',
    # Additional metadata like classifiers, keywords, etc.
)