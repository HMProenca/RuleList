from setuptools import setup

setup(
    name='rulelist',
    version='0.0.0',
    packages=['tests', 'tests.mdl', 'tests.data', 'tests.util', 'tests.search', 'tests.search.beam',
              'tests.rulelistmodel', 'tests.rulelistmodel.categoricalmodel', 'rulelist', 'rulelist.mdl',
              'rulelist.search', 'rulelist.search.beam', 'rulelist.search.preminedpatterns',
              'rulelist.measures', 'rulelist.datastructure', 'rulelist.datastructure.attribute',
              'rulelist.rulelistmodel', 'rulelist.rulelistmodel.gaussianmodel',
              'rulelist.rulelistmodel.categoricalmodel'],
    url='https://github.com/HMProenca/RuleLists',
    license='MIT License',
    author='Hugo Proenca',
    author_email='hugo.manuel.proenca@gmail.com',
    description='Generates rule lists for prediction and data mining.'
)
