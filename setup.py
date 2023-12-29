from setuptools import setup, find_packages


def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README


with open("requirements.txt", encoding="utf8") as f:
    required = f.read().splitlines()


setup(
    name='python_modules',
    version='0.0.1',
    description='Collection of Python modules for EDA and ML',
    long_description=readme(),
    author='Ioannis Kyriakos',
    author_email='ioannis.kyriakos@gmail.com',
    packages=find_packages(),
    install_requires=[
        'seaborn>=0.13.0', # General requirement
        'matplotlib>=3.8.0',
        'pandas>=2.1.1',
        'numpy>=1.26.2',
        'requests>=2.31.0',
        'scipy>=1.11.4',
        'scikit-learn>=1.3.0'
    ],
    extras_require={
        'kaggle': [
            # Specifying exact versions required for Kaggle environment - not tested only DEV
            'seaborn>=0.12.0,<0.13.1',
            'matplotlib>=3.7.4,<3.8.1',
            'numpy>=1.22.3,<1.26.3',
            'pandas>=2.0.3,<2.1.2',
            'scikit-learn>=1.2.2,<1.3.1',
            'scipy>=1.8.0,<1.11.5'
        ]
    },
    classifiers=[
        'Development Status :: Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    license='MIT',
    url='https://github.com/KyriakosJiannis/python_modules',
)
