"""
Examples
========

The following example uses a simple multi-target data set (generated with
:download:`generate_multitarget.py <code/generate_multitarget.py>`) to show
some basic functionalities (part of
:download:`multitarget.py <code/multitarget.py>`).

.. literalinclude:: code/multitarget.py
    :lines: 1-6

Multi-target learners can build prediction models (classifiers)
which then predict (multiple) class values for a new instance (continuation of
:download:`multitarget.py <code/multitarget.py>`):

.. literalinclude:: code/multitarget.py
    :lines: 8-

"""

from pkg_resources import resource_filename
def datasets():
    yield ('multitarget', resource_filename(__name__, 'datasets'))


import Orange
# Other algorithms which also work with multitarget data
from Orange.regression import earth

# Multi-target algorithms
import tree
import chain
import binary
import neural
import scoring
import pls
