import setuptools

with open('README.md') as f:
    long_description = f.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='rulelist',
    version='0.0.0',
    description='Learn rule lists from data for classification, regression or subgroup discovery',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='https://github.com/HMProenca/RuleList',
    license='MIT License',
    author='Hugo Proenca',
    author_email='hugo.manuel.proenca@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)