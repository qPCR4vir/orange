"""

.. index:: Clustering Tree Learner

***************************************
Clustering Tree Learner (``tree``)
***************************************

:obj:`ClusteringTreeLearner` is an implementation of classification and regression
trees, based on the :obj:`SimpleTreeLearner`. It is implemented in C++ for speed and low memory usage.
Clustering trees work by splitting the data into clusters based on attributes. The attribute provides the optimal split based on a measure, 
the default used in this implementation is the Euclidean distance between the centroids of clusters, which we try to maximize. Additional measures
are implemented, more information on them can be found in the parameter description.

The implementation is based on the article by Blockeel et al. [1]_

:obj:`ClusteringTreeLearner` was developed for speeding up the construction
of random forests, but can also be used as a standalone tree learner.

.. class:: ClusteringTreeLearner

    .. attribute:: min_majority

        Minimal proportion of the majority class value each of the class variables has to reach
        to stop induction (only used for classification). 

    .. attribute:: min_MSE

        Minimal mean squared error each of the class variables has to reach
        to stop induction (only used for regression). 

    .. attribute:: min_instances

        Minimal number of instances in leaves. Instance count is weighed.

    .. attribute:: max_depth

        Maximal depth of tree.

    .. attribute:: method

        The method used when chosing attributes while building the learner. The parameters should be supplied as either an integer (from 0 to 3) 
        or with Orange.multitarget.tree. followed by the name of the measure (as shown in the examples below). Possible choices are:

            * inter_distance (default) - Euclidean distance between centroids of clusters
            * intra_distance - average Euclidean distance of each member of a cluster to the centroid of that cluster
            * silhouette - silhouette (http://en.wikipedia.org/wiki/Silhouette_(clustering)) measure calculated with euclidean distances
            * gini_index - calculates the Gini-gain index, should be used with class variables with nominal values

    .. attribute:: skip_prob

        At every split an attribute will be skipped with probability ``skip_prob``.
        Useful for building random forests.

    .. attribute:: random_generator
        
        Provide your own :obj:`Orange.misc.Random`.

Examples
========

:obj:`ClusteringTreeLearner` can be used on its own or in a random forest, below are
examples of usage.


.. literalinclude:: code/clustering_tree.py



References
============
.. [1] H. Blockeel, L. De Raedt, and J. Ramon, "Top-Down Induction of Clustering Trees", 
        In Proceedings of the Fifteenth International Conference on Machine Learning (ICML '98), 55-63, 1998.

"""



from Orange.core import ClusteringTreeLearner, ClusteringTreeClassifier

#distance methods for easier access
inter_distance = 0
intra_distance = 1
silhouette = 2
gini_index = 3


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('multitarget-synthetic')
    print 'Actual classes:\n', data[0].get_classes()
    
    majority = Orange.classification.majority.MajorityLearner()
    mt_majority = Orange.multitarget.binary.BinaryRelevanceLearner(learner=majority)
    c_mtm = mt_majority(data)
    print 'Majority predictions:\n', c_mtm(data[0])

    mt_tree = ClusteringTreeLearner()
    c_mtt = mt_tree(data)
    print 'Multi-target Tree predictions:\n', c_mtt(data[0])
