Installation Guide
==================

This installation guide provides step-by-step instructions for setting up the project environment, installing dependencies, and generating the HTML documentation. Follow these sections to get started.

Installation with setup.py
--------------------------

To install this project on your system, you will need Python installed. It's recommended to use a virtual environment for Python projects to manage dependencies effectively.

1. **Clone the Project**

   First, clone the project repository from GitHub (or your preferred source) to your local machine:

   .. code-block:: bash

      git clone https://github.com/yourusername/yourprojectname.git
      cd yourprojectname

2. **Create a Virtual Environment**

   Before installing the project, create a new virtual environment in the project directory:

   .. code-block:: bash

      python -m venv venv

   Activate the virtual environment:

   **On Windows:**

   .. code-block:: bash

      .\\venv\\Scripts\\activate

   **On Unix or MacOS:**

   .. code-block:: bash

      source venv/bin/activate

3. **Install with setup.py**

   Install the project and its Python dependencies using the `setup.py` file:

   .. code-block:: bash

      python setup.py install

Creating venv and Install requirements.txt
------------------------------------------

If the project has additional dependencies listed in a `requirements.txt` file, you can install these after activating your virtual environment:

.. code-block:: bash

   pip install -r requirements.txt

This command installs all the Python packages listed in `requirements.txt`, ensuring that your environment has all the required dependencies.

Create HTML Documentation with `make html`
-------------------------------------------

Once the project is installed and all dependencies are in place, you can generate the HTML documentation using Sphinx:

1. **Navigate to the Docs Directory**

   Change directory to the `docs` folder where the Sphinx configuration file (`conf.py`) is located:

   .. code-block:: bash

      cd docs

2. **Build the Documentation**

   Use the `make` command to build the HTML documentation:

   .. code-block:: bash

      make html

   This command instructs Sphinx to generate the documentation and place the output files in the `_build/html` directory within the `docs` folder.

3. **View the Documentation**

   After successfully building the documentation, open the `index.html` file in the `_build/html` directory with your web browser to view the HTML documentation:

   .. code-block:: bash

      # On Windows, you might use:
      start _build/html/index.html

      # On Unix or MacOS, you might use:
      open _build/html/index.html

Congratulations! You have successfully set up and built the HTML documentation for the project.
