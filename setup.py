#!/usr/bin/env python3
"""matrixor setup.py.

This file details modalities for packaging the matrixor application.
"""

from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='matrixor',
    description='Matrix transformator in Python',
    author=' Alexandre Kabbach',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.2.0',
    url='https://github.com/akb89/matrixor',
    download_url='https://github.com/akb89/matrixor',
    license='MIT',
    keywords=['matrix', 'linear transformation'],
    platforms=['any'],
    packages=['matrixor', 'matrixor.utils',
              'matrixor.logging', 'matrixor.exceptions'],
    package_data={'matrixor': ['logging/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'matrixor = matrixor.main:main'
        ],
    },
    install_requires=['pyyaml>=4.2b1', 'einsumt==0.9.1', 'scipy==1.2.0'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
