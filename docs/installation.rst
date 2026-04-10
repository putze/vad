Installation
============

This page describes how to install the project and its optional components.

Requirements
------------

The project requires:

- Python 3.10, 3.11, or 3.12
- pip
- A virtual environment (recommended)


Setup
-----

Clone the repository and create a virtual environment:

.. code-block:: bash

   git clone https://github.com/putze/vad.git
   cd vad

   python -m venv .venv
   source .venv/bin/activate

Upgrade pip and install the project:

.. code-block:: bash

   python -m pip install --upgrade pip
   pip install -e .

This installs the core dependencies required for training and inference.

Optional Dependencies
---------------------

The project defines several optional dependency groups.

Development
~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[dev]"

Includes:

- pytest, pytest-cov
- ruff, mypy
- bandit, pip-audit, vulture

Documentation
~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[docs]"

Includes:

- sphinx
- furo
- myst-parser

Demo
~~~~

.. code-block:: bash

   pip install -e ".[demo]"

Includes:

- streamlit
- matplotlib
- plotly
- pandas

Baselines
~~~~~~~~~

.. code-block:: bash

   pip install -e ".[baseline]"

Includes:

- webrtcvad

Full Installation
-----------------

For a complete setup:

.. code-block:: bash

   pip install -e ".[dev,docs,demo,baseline]"

Command-Line Tools
------------------

The following commands are installed:

- ``vad-train``
- ``vad-infer-offline``
- ``vad-stream-file``
- ``vad-compare-models``
- ``vad-demo``

Build Documentation
-------------------

To build the documentation locally:

.. code-block:: bash

   make -C docs html

Output is generated in:

.. code-block:: text

   docs/_build/html
