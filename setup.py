from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='TimeWeaver',  # Package name
    version='0.1.1',  # Package version
    author='Richard Scholz',  # Your name
    author_email='richardscholz1@gmx.de',  # Your email address
    description='Python Package for automated multivariate Time Series imputation',  # Short package description
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # README.md content type
    url='https://github.com/rscholz98/TimeWeaver',  # Package URL
    package_dir={'': 'src'},  # Specify the source directory
    packages=find_packages(where='src'),  # Automatically find packages in src
    install_requires=requirements,  # Dependencies from requirements.txt
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  # Minimum Python version requirement
)

# Setup local package 
# Start with $ pip install -e . 