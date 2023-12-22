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
        'seaborn>=0.12.2',
        'matplotlib>=3.7.4',
        'pandas>=2.0.2',
        'numpy>=1.24.3',
        'requests>=2.31.0',
        'scipy>=1.7.3',
        'scikit-learn>=1.2.2'

    ],
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
