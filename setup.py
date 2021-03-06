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
    version='1.0.7',
    url='https://github.com/akb89/matrixor',
    download_url='https://github.com/akb89/matrixor',
    license='MIT',
    keywords=['matrix', 'linear transformation'],
    platforms=['any'],
    packages=['matrixor'],
    entry_points={
        'console_scripts': [
            'matrixor = matrixor.main:main'
        ],
    },
    install_requires=['numpy==1.19.0'],
    classifiers=['Development Status :: 5 - Production/Stable ',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    zip_safe=False,
)
