import setuptools

with open('README.md') as f:
    long_description = f.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name= 'rulelist',
    version='0.2.1',
    description='Learn rule lists from data for classification, regression or subgroup discovery',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='https://github.com/HMProenca/RuleList',
    license='MIT License',
    author='Hugo Proenca',
    author_email='hugo.manuel.proenca@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)