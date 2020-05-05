from setuptools import setup, find_namespace_packages

setup(name='pyarena',
      version='0.1',
      description='A python library of robotic algorithms',
      author='Praveen R. Jain and Romulo T. Rodrigues',
      license='GPLv3',
      packages=find_namespace_packages(include=['pyarena.*']),
      zip_safe=False)
