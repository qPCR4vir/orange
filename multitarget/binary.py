"""
.. index:: Binary Relevance Learner

***************************************
Binary Relevance Learner (``binary``)
***************************************


.. index:: Multi-target Binary Relevance Learner
.. autoclass:: Orange.multitarget.binary.BinaryRelevanceLearner
    :members:
    :show-inheritance:

.. index:: Multi-target Binary Relevance Classifier
.. autoclass:: Orange.multitarget.binary.BinaryRelevanceClassifier
    :members:
    :show-inheritance:

"""

import Orange.core as orange
import Orange
import copy
from operator import add

class BinaryRelevanceLearner(orange.Learner):
    """
    Creates a standard single class learner for each class variable. Binary relevance assumes independance of class variables.

    :param learner: A single class learner.
    :type learner: :class:`Orange.core.Learner`

    :param callback: a function to be called after every iteration of
            induction of classifier. The call returns a parameter
            (from 0.0 to 1.0) that provides an estimate
            of completion of the learning progress.

    :param name: learner name.
    :type name: string

    :rtype: :class:`Orange.multitarget.BinaryRelevanceLearner` or 
            :class:`Orange.multitarget.BinaryRelevanceCLassifier`

    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)
        if data:   
            self.__init__(**kwargs)
            return self(data,weight)
        else:
            return self

    def __init__(self, learner=None, name="Binary Relevance", callback=None, **kwargs):
        self.name = name
        self.callback = callback

        if not learner:
            raise TypeError("Wrong specification, learner not defined")
        else:
            self.learner = learner
        self.__dict__.update(kwargs)     

    def __call__(self, instances, weight=0):
        """
        Construct learners from the given table of data instances.
        
        :param instances: data for learning.
        :type instances: class:`Orange.data.Table`

        :param weight: weight.
        :type weight: int

        :rtype: :class:`Orange.multitarget.BinaryRelevanceClassifier`
        """

        if not instances.domain.class_vars: raise Exception('No classes defined.')

        m = len(instances.domain.class_vars)

        classifiers = [None for _ in xrange(m)]
        domains = [Orange.data.Domain(instances.domain.attributes, cv) \
                   for cv in instances.domain.class_vars]

        for i in range(m):
            classifiers[i] = self.learner(Orange.data.Table(domains[i], instances), weight)
            if self.callback:
                self.callback((i + 1.0) / m)

        return BinaryRelevanceClassifier(classifiers=classifiers, domains=domains, name=self.name)


class BinaryRelevanceClassifier(orange.Classifier):
    """
    Uses the classifiers induced by the :obj:`BinaryRelevanceLearner`. An input
    instance is classified into the class with the most frequent vote.
    However, this implementation returns the averaged probabilities from
    each of the trees if class probability is requested.
    
    :param classifiers: a list of classifiers.
    :type classifiers: list of  :class:`Orange.core.Learner`
        
    :param domains: the domains used by learners.
    :type domain: list of :class:`Orange.data.Domain`
    
    :param name: name of the classifier.
    :type name: string

    """

    def __init__(self, classifiers, domains, name):
        self.classifiers = classifiers
        self.domains = domains
        self.name = name   

    def __call__(self, instance, result_type = orange.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """

        predictions = [c(Orange.data.Instance(dom, instance), result_type) \
                       for c, dom in zip(self.classifiers, self.domains)]

        return zip(*predictions) if result_type == Orange.core.GetBoth \
               else predictions

    def __reduce__(self):
        return type(self), (self.classifiers, self.domains, self.name), dict(self.__dict__)

		
if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    data = Orange.data.Table('multitarget:bridges.tab')
    
    l1 = BinaryRelevanceLearner(learner = Orange.classification.tree.SimpleTreeLearner)
    l2 = BinaryRelevanceLearner(learner = Orange.classification.bayes.NaiveLearner)
    l3 = BinaryRelevanceLearner(learner = Orange.classification.majority.MajorityLearner)

    res = Orange.evaluation.testing.cross_validation([l1,l2,l3],data)

    scores = Orange.multitarget.scoring.mt_average_score(res,Orange.evaluation.scoring.RMSE)

    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)
