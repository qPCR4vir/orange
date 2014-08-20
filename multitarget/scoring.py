"""
.. index:: Multi-target Scoring


***************************************
Multi-target Scoring (``scoring``)
***************************************

:doc:`Multi-target <Orange.multitarget>` classifiers predict values for
multiple target classes. They can be used with standard
:obj:`~Orange.evaluation.testing` procedures (e.g.
:obj:`~Orange.evaluation.testing.Evaluation.cross_validation`), but require
special scoring functions to compute a single score from the obtained
:obj:`~Orange.evaluation.testing.ExperimentResults`.
Since different targets can vary in importance depending on the experiment,
some methods have options to indicate this e.g. through weights or customized
distance functions. These can also be used for normalization in case target
values do not have the same scales.

.. autofunction:: mt_flattened_score
.. autofunction:: mt_average_score

The whole procedure of evaluating multi-target methods and computing
the scores (RMSE errors) is shown in the following example
(:download:`mt-evaluate.py <code/mt-evaluate.py>`). Because we consider
the first target to be more important and the last not so much we will
indicate this using appropriate weights.

.. literalinclude:: code/mt-evaluate.py

Which outputs::

    Weighted RMSE scores:
        Majority    0.8228
      Clust Tree    0.4528
             PLS    0.3021
           Earth    0.2880

Two more accuracy measures based on the article by Zaragoza et al._[1]; applicable to discrete classes:

Global accuracy (accuracy per example) over d-dimensional class variable:

.. autofunction:: mt_global_accuracy

Mean accuracy (accuracy per class or per label) over d class variables: 

.. autofunction:: mt_mean_accuracy   

References
============
.. [1] Zaragoza, J.H., Sucar, L.E., Morales, E.F.,Bielza, C., Larranaga, P.  (2011). 'Bayesian Chain Classifiers for Multidimensional Classification',
         Proc. of the International Joint Conference on Artificial Intelligence (IJCAI-2011),  pp:2192-2197.




"""

import Orange
from Orange import statc, corn
from collections import Iterable
from Orange.utils import deprecated_keywords, deprecated_function_name, \
    deprecation_warning, environ

def mt_average_score(res, score, weights=None):
    """
    Compute individual scores for each target and return the (weighted) average.

    One method can be used to compute scores for all targets or a list of
    scoring methods can be passed to use different methods for different
    targets. In the latter case, care has to be taken if the ranges of scoring
    methods differ.
    For example, when the first target is scored from -1 to 1 (1 best) and the
    second from 0 to 1 (0 best), using `weights=[0.5,-1]` would scale both
    to a span of 1, and invert the second so that higher scores are better.

    :param score: Single-target scoring method or a list of such methods
                  (one for each target).
    :param weights: List of real weights, one for each target,
                    for a weighted average.

    """
    if not len(res.results):
        raise ValueError, "Cannot compute the score: no examples."
    if res.number_of_learners < 1:
        return []
    n_classes = len(res.results[0].actual_class)
    if weights is None:
        weights = [1.] * n_classes
    if not isinstance(score, Iterable):
        score = [score] * n_classes
    elif len(score) != n_classes:
        raise ValueError, "Number of scoring methods and targets do not match."
    # save original classes
    clsss = [te.classes for te in res.results]
    aclsss = [te.actual_class for te in res.results]
    probss = [te.probabilities if te.probabilities else None for te in res.results]
    cls_vals = res.class_values if res.class_values else None

    # compute single target scores
    single_scores = []
    for i in range(n_classes):
        for te, clss, aclss, probs in zip(res.results, clsss, aclsss, probss):
            te.classes = [cls[i] for cls in clss]
            te.actual_class = aclss[i]
            te.probabilities = [prob[i] for prob in probs] if probs else None
        res.class_values = cls_vals[i] if cls_vals else None

        single_scores.append(score[i](res))
    # restore original classes
    for te, clss, aclss, probs in zip(res.results, clsss, aclsss, probss):
        te.classes = clss
        te.actual_class = aclss
        te.probabilities = probs
    res.class_values = cls_vals

    return [sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        for scores in zip(*single_scores)]

def mt_flattened_score(res, score):
    """
    Flatten (concatenate into a single list) the predictions of multiple
    targets and compute a single-target score.
    
    :param score: Single-target scoring method.
    """
    res2 = Orange.evaluation.testing.ExperimentResults(res.number_of_iterations,
        res.classifier_names, class_values=res.class_values,
        weights=res.weights, classifiers=res.classifiers, loaded=res.loaded,
        test_type=Orange.evaluation.testing.TEST_TYPE_SINGLE, labels=res.labels)
    for te in res.results:
        for i, ac in enumerate(te.actual_class):
            te2 = Orange.evaluation.testing.TestedExample(
                iteration_number=te.iteration_number, actual_class=ac)
            for c, p in zip(te.classes, te.probabilities):
                te2.add_result(c[i], p[i])
            res2.results.append(te2)
    return score(res2)

def mt_global_accuracy(res):
    """
    :math:`Acc = \\frac{1}{N}\\sum_{i=1}^{N}\\delta(\\mathbf{c_{i}'},\\mathbf{c_{i}}) \\newline`
	
    :math:`\\delta (\\mathbf{c_{i}'},\\mathbf{c_{i}} )=\\left\\{\\begin{matrix}1:\\mathbf{c_{i}'}=\\mathbf{c_{i}}\\\\ 0: otherwise\\end{matrix}\\right.`
    """
    results = []
    for l in xrange(res.number_of_learners):
        n_results = len(res.results)
        n_correct = 0.

        for r in res.results:
            if list(r.classes[l]) == r.actual_class:
                n_correct+=1

        results.append(n_correct/n_results)
    return results


def mt_mean_accuracy(res):
    """
    :math:`\\overline{Acc_{d}} = \\frac{1}{d}\\sum_{j=1}^{d}Acc_{j} = \\frac{1}{d}\\sum_{j=1}^{d} \\frac{1}{N}\\sum_{i=1}^{N}\\delta(c_{ij}',c_{ij} ) \\newline`
	
    :math:`\\delta (c_{ij}',c_{ij} )=\\left\\{\\begin{matrix}1:c_{ij}'=c_{ij}\\\\ 0: otherwise\\end{matrix}\\right.`
    """
    results = []
    for l in xrange(res.number_of_learners):
        n_classes = len(res.results[0].actual_class)
        n_results = len(res.results)
        n_correct = 0.

        for r in res.results:
            for i in xrange(n_classes):
                if r.classes[l][i] == r.actual_class[i]:
                    n_correct+=1
        results.append(n_correct/n_classes/n_results)
    return results


################################################################################
if __name__ == "__main__":
    pass
