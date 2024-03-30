from setuptools import setup, find_packages

setup(
    name='TimeWeaver',  # Package name
    version='0.1.7.31',  # Package version
    author='Richard Scholz',  # Your name
    author_email='richardscholz1@gmx.de',  # Your email address
    description='Python Package for automated multivariate Time Series imputation',  # Short package description
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # README.md content type
    url='https://github.com/rscholz98/TimeWeaver',  # Package URL
    package_dir={'': 'src'},  # Specify the source directory
    packages=find_packages(where='src'),  # Automatically find packages in src
    include_package_data=True,
    install_requires=[
        'scipy==1.12.0',
        'pandas==2.2.0',
        'tsfresh==0.20.2',
        'tsfel==0.1.6',
        'plotly==5.18.0',
        'nbformat==5.9.2',
        'seaborn==0.13.2',
        'kaleido==0.2.1',
        'sphinx==5.0.1',
        'sphinx-rtd-theme==1.0.0',
        'sphinx-design==0.5.0',
        'pyreadr==0.5.0',
        'tabulate==0.9.0',
        'twine==5.0.0'
    ], 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimum Python version requirement
)

# Setup local package 
# Start with $ pip install -e .
# python setup.py sdist
# twine upload dist/TimeWeaver-0.1.7.4.tar.gz