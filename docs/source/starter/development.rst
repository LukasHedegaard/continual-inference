####################
Development setup
####################

Clone repository:

.. code-block:: bash

    git clone https://github.com/lukashedegaard/continual-inference.git
    cd ride

Install extended dependencies:

.. code-block:: bash

    pip install -e .[build,dev,docs]
    npm install -g katex

Run tests:

.. code-block:: bash

    make test

Build docs

.. code-block:: bash

    cd docs
    make html

Build and publish to TestPyPI:

.. code-block:: bash

    make clean
    make testbuild
    make testpublish

Build and publish to PyPI:

.. code-block:: bash

    make clean
    make build
    make publish