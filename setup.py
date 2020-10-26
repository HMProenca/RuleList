from setuptools import setup

setup(
    name='rulelists',
    version='0.0.0',
    packages=['tests', 'tests.mdl', 'tests.data', 'tests.util', 'tests.search', 'tests.search.beam',
              'tests.rulelistmodel', 'tests.rulelistmodel.categoricalmodel', 'mdlrulelist', 'mdlrulelist.mdl',
              'mdlrulelist.search', 'mdlrulelist.search.beam', 'mdlrulelist.search.preminedpatterns',
              'mdlrulelist.measures', 'mdlrulelist.datastructure', 'mdlrulelist.datastructure.attribute',
              'mdlrulelist.rulelistmodel', 'mdlrulelist.rulelistmodel.gaussianmodel',
              'mdlrulelist.rulelistmodel.categoricalmodel'],
    url='https://github.com/HMProenca/RuleLists',
    license='MIT License',
    author='Hugo Proenca',
    author_email='hugo.manuel.proenca@gmail.com',
    description='Generates rule lists for prediction and data mining.'
)
