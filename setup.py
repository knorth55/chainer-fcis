from setuptools import find_packages
from setuptools import setup


version = '2.1.0'


setup(
    name='fcis',
    version=version,
    packages=find_packages(),
    scripts=['scripts/convert_model.py'],
    install_requires=open('requirements.txt').readlines(),
    description='Chainer Implementation of FCIS',
    long_description=open('README.md').read(),
    author='Shingo Kitagawa',
    author_email='shingogo.5511@gmail.com',
    url='https://github.com/knorth55/chainer-fcis',
    license='MIT',
    keywords='machine-learning',
)
