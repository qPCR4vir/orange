"""
.. index:: Classifier Chain Learner

***************************************
Classifier Chain Learner (``chain``)
***************************************


.. index:: Multi-target Classifier Chain Learner
.. autoclass:: Orange.multitarget.chain.ClassifierChainLearner
    :members:
    :show-inheritance:

.. index:: Multi-target Classifier Chain Classifier
.. autoclass:: Orange.multitarget.chain.ClassifierChain
    :members:
    :show-inheritance:

***************************************
Ensemble Classifier Chain Learner
***************************************


.. index:: Multi-target Ensemble Classifier Chain Learner
.. autoclass:: Orange.multitarget.chain.EnsembleClassifierChainLearner
    :members:
    :show-inheritance:

.. index:: Multi-target Ensemble Classifier Chain Classifier
.. autoclass:: Orange.multitarget.chain.EnsembleClassifierChain
    :members:
    :show-inheritance:

"""

import Orange.core as orange
import Orange
import random
import copy
from operator import add

class ClassifierChainLearner(orange.Learner):
    """
    Expands single class classification techniques into multi-target classification techniques by chaining the classification
    data. A learner is constructed for each of the class variables in a random or given order. The data for each learner are
    the features extended by all previously classified variables. This chaining passes the class informationd between
    classifiers and allows class correlations to be taken into account.
    TODO: cite weka source?

    :param learner: A single class learner that will be extended by chaining.
    :type learner: :class:`Orange.core.Learner`

    :param rand: random generator used in bootstrap sampling. If None (default), 
        then ``random.Random(42)`` is used.

    :param callback: a function to be called after every iteration of
            induction of classifier. The call returns a parameter
            (from 0.0 to 1.0) that provides an estimate
            of completion of the learning progress.

    :param name: learner name.
    :type name: string

    :rtype: :class:`Orange.multitarget.chain.ClassifierChain` or 
            :class:`Orange.multitarget.chain.ClassifierChainLearner`

    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)
        if data:   
            self.__init__(**kwargs)
            return self(data,weight)
        else:
            return self

    def __init__(self, learner=None, name="Classifier Chain", rand=None, callback=None, class_order=None, actual_values=True, **kwargs):
        self.name = name
        self.rand = rand
        self.callback = callback
        self.class_order = class_order
        self.actual_values = actual_values

        if not learner:
            raise TypeError("Wrong specification, learner not defined")
        else:
            self.learner = learner

        if not self.rand:
            self.rand = random.Random(42)
        self.__dict__.update(kwargs)     

        self.randstate = self.rand.getstate()
           
    def __call__(self, instances, weight=0, class_order=None):
        """
        Learn from the given table of data instances.
        
        :param instances: data for learning.
        :type instances: class:`Orange.data.Table`

        :param weight: weight.
        :type weight: int

        :param class_order: list of descriptors of class variables
        :type class_order: list of :class:`Orange.feature.Descriptor`

        :rtype: :class:`Orange.multitarget.chain.ClassifierChain`
        """

        instances = Orange.data.Table(instances.domain, instances) # bypasses ownership

        self.rand.setstate(self.randstate) 
        n = len(instances)
        m = len(instances.domain.class_vars)

        classifiers = [None for _ in xrange(m)]
        domains = [None for _ in xrange(m)]
        orig_domain = copy.copy(instances.domain)

        if self.class_order and not class_order:
            class_order = self.class_order
        elif not class_order:
            class_order = [cv for cv in instances.domain.class_vars]
            self.rand.shuffle(class_order)

        learner = self.learner

        for i in range(m):
            # sets one of the class_vars as class_var
            instances.pick_class(class_order[i])            

            # save domains for classification
            domains[i] = Orange.data.Domain([d for d in instances.domain])

            #overide lingering class_vars
            data = Orange.data.Table(domains[i], instances)

            classifiers[i] = learner(data, weight)

            if not self.actual_values:
                for j in xrange(len(instances)):
                    instances[j][-1] = classifiers[i](data[j])

            # updates domain to include class_var in features
            instances.change_domain(Orange.data.Domain(instances.domain, False, \
                class_vars=instances.domain.class_vars))

            if self.callback:
                self.callback((i + 1.0) / m)

        return ClassifierChain(classifiers=classifiers, class_order=class_order, domains=domains, name=self.name, orig_domain=orig_domain)


class ClassifierChain(orange.Classifier):
    """
    Uses the classifiers induced by the :obj:`ClassifierChainLearner`. An input
    instance is classified into the class with the most frequent vote.
    However, this implementation returns the averaged probabilities from
    each of the trees if class probability is requested.

    It should not be constructed manually.
    
    :param classifiers: a list of classifiers.
    :type classifiers: list of  :class:`Orange.core.Learner`
        
    :param domains: the domains used by learners.
    :type domain: list of :class:`Orange.data.Domain`
    
    :param orig_domain: the domain of the learning set.
    :type domain: :class:`Orange.data.Domain`

    :param name: name of the classifier.
    :type name: string

    """

    def __init__(self, classifiers, class_order, domains, name, orig_domain):
        self.classifiers = classifiers
        self.class_order = class_order
        self.name = name
        self.domains = domains
        self.orig_domain = orig_domain

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

        inst = [v for v in instance]
        values = {cv:None for cv in instance.domain.class_vars}
        probs = {cv:None for cv in instance.domain.class_vars}

        for i in range(len(self.class_order)):
            # add blank class for classification

            inst = Orange.data.Instance(self.domains[i],[v for v in inst] + ['?'])
  
            res = self.classifiers[i](inst, orange.GetBoth)
            values[self.class_order[i]] = res[0]
            probs[self.class_order[i]] = res[1]
            inst[-1] = res[0] # set the classified value for chaining

        # sort values and probabilites to original order
        values = [values[cv] for cv in self.orig_domain.class_vars]
        probs = [probs[cv] for cv in self.orig_domain.class_vars]

        if result_type == orange.GetValue: return tuple(values)
        elif result_type == orange.GetProbabilities: return tuple(probs)
        else: 
            return [tuple(values),tuple(probs)]

    def __reduce__(self):
        return type(self), (self.classifiers, self.class_order, self.domains, self.name, self.orig_domain), dict(self.__dict__)


class EnsembleClassifierChainLearner(orange.Learner):
    """
    Expands single class classification techniques into multi-target classification techniques by chaining the classification
    data. A learner is constructed for each of the class variables in a random or given order. The data for each learner are
    the features extended by all previously classified variables. This chaining passes the class informationd between
    classifiers and allows class correlations to be taken into account.
    TODO: cite weka source?

    :param n_chains: Number of chains to be constructed.
    :type n_chains: integer

    :param sample_size: Size (percentage) of the subset taken from original dataset.
    :type sample_size: float

    :param learner: A single class learner that will be extended by chaining.
    :type learner: :class:`Orange.core.Learner`

    :param rand: random generator used in bootstrap sampling. If None (default), 
        then ``random.Random(42)`` is used.


    :param callback: a function to be called after every iteration of
            induction of classifier. The call returns a parameter
            (from 0.0 to 1.0) that provides an estimate
            of completion of the learning progress.

    :param name: learner name.
    :type name: string

    :rtype: :class:`Orange.multitarget.chain.EnsembleClassifierChainLearner` or 
            :class:`Orange.multitarget.chain.EnsembleClassifierChain`
    """

    def __new__(cls, data=None, weight = 0, **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)
        if data:   
            self.__init__(**kwargs)
            return self(data,weight)
        else:
            return self

    def __init__(self, n_chains=50, sample_size=0.25, learner=None, actual_values=True, name="Ensemble CChain", rand=None, callback=None):
        self.n_chains = n_chains
        self.sample_size = sample_size
        self.name = name
        self.rand = rand
        self.callback = callback
        self.actual_values = actual_values

        if not learner:
            raise TypeError("Wrong specification, learner not defined")
        else:
            self.learner = learner

        if not self.rand:
            self.rand = random.Random(42)

        self.randstate = self.rand.getstate()

    def __call__(self, instances, weight=0):
        """
        Learn from the given table of data instances.
        
        :param instances: data for learning.
        :type instances: class:`Orange.data.Table`

        :param weight: weight.
        :type weight: int

        :param class_order: list of descriptors of class variables
        :type class_order: list of :class:`Orange.feature.Descriptor`

        :rtype: :class:`Orange.multitarget.chain.ClassifierChain`
        """

        ind = Orange.data.sample.SubsetIndices2(p0=1-self.sample_size)
        indices = ind(instances)
        class_order = [cv for cv in instances.domain.class_vars]
        classifiers = []

        for i in range(self.n_chains):
            #randomize chain ordering and subset of instances
            self.rand.shuffle(class_order)
            self.rand.shuffle(indices)
            data = instances.select_ref(indices,1)

            learner = ClassifierChainLearner(learner=self.learner, actual_values=self.actual_values, rand=self.rand) # TODO might work in one step

            classifiers.append(learner(data, weight, copy.copy(class_order)))

            if self.callback:
                self.callback((i+1.)/self.n_chains)

        return EnsembleClassifierChain(classifiers=classifiers, name=self.name)

class EnsembleClassifierChain(orange.Classifier):
    """
    Uses the classifiers induced by the :obj:`EnsembleClassifierChainLearner`. An input
    instance is classified into the class with the most frequent vote.
    However, this implementation returns the averaged probabilities from
    each of the trees if class probability is requested.
   
    When constructed manually, the following parameters have to
    be passed:

    :param classifiers: a list of chain classifiers to be used.
    :type classifiers: list
    
    :param name: name of the resulting classifier.
    :type name: str

    """

    def __init__(self, classifiers, name):
        self.classifiers = classifiers
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

        class_vars = instance.domain.class_vars

        # get results to avoid multiple calls
        res_both = [c(instance, orange.GetBoth) for c in self.classifiers]

        mt_prob = []
        mt_value = []

        for varn in xrange(len(class_vars)):

            class_var = class_vars[varn]
            # handle discreete class
        
            if class_var.var_type == Orange.feature.Discrete.Discrete:

                # voting for class probabilities
                if result_type == orange.GetProbabilities or result_type == orange.GetBoth:
                    prob = [0.] * len(class_var.values)
                    for r in res_both:
                        a = [x for x in r[1][varn]]
                        prob = map(add, prob, a)
                    norm = sum(prob)
                    cprob = Orange.statistics.distribution.Discrete(class_var)
                    for i in range(len(prob)):
                        cprob[i] = prob[i]/norm
                
                # voting for crisp class membership, notice that
                # this may not be the same class as one obtaining the
                # highest probability through probability voting
                if result_type == orange.GetValue or result_type == orange.GetBoth:
                    cfreq = [0] * len(class_var.values)
                    for r in res_both:
                        cfreq[int(r[0][varn])] += 1
                    index = cfreq.index(max(cfreq))
                    cvalue = Orange.data.Value(class_var, index)
            
                if result_type == orange.GetValue: mt_value.append(cvalue)
                elif result_type == orange.GetProbabilities: mt_prob.append(cprob)
                else: 
                    mt_value.append(cvalue)
                    mt_prob.append(cprob)
        
            else:
                # Handle continuous class
        
                # voting for class probabilities
                if result_type == orange.GetProbabilities or result_type == orange.GetBoth:
                    probs = [ r for r in res_both]
                    cprob = dict()
      
                    for val,prob in probs:
                        if prob[varn] != None: #no probability output
                            a = dict(prob[varn].items())
                        else:
                            a = { val[varn].value : 1. }
                        cprob = dict( (n, a.get(n, 0)+cprob.get(n, 0)) for n in set(a)|set(cprob) )
                    cprob = Orange.statistics.distribution.Continuous(cprob)
                    cprob.normalize()
                
                # gather average class value
                if result_type == orange.GetValue or result_type == orange.GetBoth:
                    values = [r[0][varn] for r in res_both]
                    cvalue = Orange.data.Value(class_var, sum(values) / len(self.classifiers))
            
                if result_type == orange.GetValue: mt_value.append(cvalue)
                elif result_type == orange.GetProbabilities: mt_prob.append(cprob)
                else: 
                    mt_value.append(cvalue)
                    mt_prob.append(cprob)
                    
        if result_type == orange.GetValue: return tuple(mt_value)
        elif result_type == orange.GetProbabilities: return tuple(mt_prob)
        else: 
            return [tuple(mt_value),tuple(mt_prob)]

    def __reduce__(self):
        return type(self), (self.classifiers, self.class_order, self.domains, self.name), dict(self.__dict__)


if __name__ == '__main__':

    print "STARTED"
    import time
    global_timer = time.time()
    data = Orange.data.Table('multitarget:bridges.tab')

    cl1 = ClassifierChainLearner(learner = Orange.classification.tree.SimpleTreeLearner, name="CChain - Tree")
    cl2 = ClassifierChainLearner(learner = Orange.classification.majority.MajorityLearner, name="CChain - Maj")
    cl3 = EnsembleClassifierChainLearner(learner = Orange.classification.tree.SimpleTreeLearner, n_chains=50, sample_size=0.25, name="Ensemble CC - Tree")
    cl4 = EnsembleClassifierChainLearner(learner = Orange.classification.majority.MajorityLearner, n_chains=50, sample_size=0.25, name="Ensemble CC - Maj")
    
    learners = [cl1,cl2,cl3,cl4]

    results = Orange.evaluation.testing.cross_validation(learners, data)


    print "%18s  %6s  %8s  %8s" % ("Learner    ", "LogLoss", "Mean Acc", "Glob Acc")
    for i in range(len(learners)):
        print "%18s  %1.4f    %1.4f    %1.4f" % (learners[i].name,
        Orange.multitarget.scoring.mt_average_score(results, Orange.evaluation.scoring.logloss)[i],
        Orange.multitarget.scoring.mt_mean_accuracy(results)[i],
        Orange.multitarget.scoring.mt_global_accuracy(results)[i])



    print "--DONE %.2f --" % (time.time()-global_timer)
