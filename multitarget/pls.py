"""
.. index:: PLS Classification Learner


***************************************
PLS Classification Learner (``pls``)
***************************************


.. index:: PLS CLassification Learner
.. autoclass:: Orange.multitarget.pls.PLSClassificationLearner
    :members:
    :show-inheritance:

.. index:: PLS Classifier
.. autoclass:: Orange.multitarget.pls.PLSClassifier
    :members:
    :show-inheritance:

"""


from Orange.regression.pls import PLSRegressionLearner, PLSRegression
import Orange

class PLSClassificationLearner(Orange.classification.Learner):
    """
    Expands and wraps :class:`Orange.regression.PLSRegressionLearner` to support classification. Multi-classes are 
    expanded with :class:`Orange.data.continuization.DomainContinuizer`.

    :rtype: :class:`Orange.multitarget.psl.PLSClassifier` or
    :rtype: :class:`Orange.multitarget.psl.PLSClassificationLearner`

    """
    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)

        if data is None:   
            return self
        else:
            self.__init__(**kwargs)
            return self(data,weight)

    def __call__(self,data,weight=0, **kwargs):
        """
        Learn from the given table of data instances.
        
        :param instances: data for learning.
        :type instances: class:`Orange.data.Table`

        :param weight: weight.
        :type weight: int

        :rtype: :class:`Orange.multitarget.psl.PLSClassifier`
        """
        cont = Orange.data.continuization.DomainContinuizer(multinomial_treatment = Orange.data.continuization.DomainContinuizer.NValues)
        pls = Orange.regression.pls.PLSRegressionLearner(data, weight, continuizer = cont,  **kwargs)

        cvals = [len(cv.values) if len(cv.values) > 2 else 1 for cv in data.domain.class_vars]
        cvals = [0] + [sum(cvals[0:i]) for i in xrange(1, len(cvals) + 1)]

        return PLSClassifier(classifier=pls, domain=data.domain, cvals=cvals)

class PLSClassifier():
    """
    Uses the classifier induced by :class:`Orange.multitarget.psl.PLSClassificationLearner`.

    """
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type=Orange.core.GetValue):
        """
        :param instance: instance to be classified.
        :type instance: :class:`Orange.data.Instance`
        
        :param result_type: :class:`Orange.classification.Classifier.GetValue` or \
              :class:`Orange.classification.Classifier.GetProbabilities` or
              :class:`Orange.classification.Classifier.GetBoth`
        
        :rtype: :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both
        """
        res = self.classifier(example, 1)
        mt_prob = []
        mt_value = []

        for cls in xrange(len(self.domain.class_vars)):
            if self.cvals[cls + 1] - self.cvals[cls] > 2:
                cprob = Orange.statistics.distribution.Discrete([p.keys()[0] for p in res[self.cvals[cls]:self.cvals[cls+1]]])
                cprob.normalize()
            else:
                r = res[self.cvals[cls]].keys()[0]
                cprob = Orange.statistics.distribution.Discrete([1.0 - r, r])

            mt_prob.append(cprob)
            mt_value.append(Orange.data.Value(self.domain.class_vars[cls], cprob.values().index(max(cprob))))

        if result_type == Orange.core.GetValue: return tuple(mt_value)
        elif result_type == Orange.core.GetProbabilities: return tuple(mt_prob)
        else: 
            return [tuple(mt_value),tuple(mt_prob)]

if __name__ == '__main__':
    import time
    print "STARTED"
    global_timer = time.time()

    l = Orange.multitarget.pls.PLSClassificationLearner()

    data = Orange.data.Table('multitarget:emotions.tab')
    res = Orange.evaluation.testing.cross_validation([l],data, 3)
    scores = Orange.multitarget.scoring.mt_average_score(res,Orange.evaluation.scoring.RMSE)

    for i in range(len(scores)):
        print res.classifierNames[i], scores[i]

    data = Orange.data.Table('multitarget:bridges.tab')
    res = Orange.evaluation.testing.cross_validation([l],data, 3)
    scores = Orange.multitarget.scoring.mt_average_score(res,Orange.evaluation.scoring.RMSE)

    for i in range(len(scores)):
       print res.classifierNames[i], scores[i]

    print "--DONE %.2f --" % (time.time()-global_timer)