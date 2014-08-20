"""
<icon>icons/TestMTLearners.png</icon>
<name>Test Multitarget Learners</name>
</description>A widget for scoring the performance of learning algorithms
on multitarget domains</description>
<category>Multitarget</category>
<priority>2000</priority>
"""

import Orange
import Orange.multitarget.scoring
from Orange.evaluation import testing, scoring
from Orange.data import sample

from OWWidget import *
import OWGUI

from orngWrap import PreprocessedLearner

from OWTestLearners import OWTestLearners, Score, Learner


def avg_logloss(res):
    return Orange.multitarget.scoring.mt_average_score(
                res, Orange.evaluation.scoring.logloss)


def flat_logloss(res):
    return Orange.multitarget.scoring.mt_flattened_score(
                res, Orange.evaluation.scoring.logloss)


def avg_is(res):
    return Orange.multitarget.scoring.mt_average_score(
                res, Orange.evaluation.scoring.IS)


def flat_is(res):
    return Orange.multitarget.scoring.mt_flattened_score(
                res, Orange.evaluation.scoring.IS)


def avg_bs(res):
    return Orange.multitarget.scoring.mt_average_score(
                res, Orange.evaluation.scoring.Brier_score)


def flat_bs(res):
    return Orange.multitarget.scoring.mt_flattened_score(
                res, Orange.evaluation.scoring.Brier_score)


def avg_rmse(res):
    return Orange.multitarget.scoring.mt_average_score(
                res, Orange.evaluation.scoring.RMSE)


def flat_rmse(res):
    return Orange.multitarget.scoring.mt_flattened_score(
                res, Orange.evaluation.scoring.RMSE)


def is_discrete(var):
    return isinstance(var, Orange.feature.Discrete)


def is_continuous(var):
    return isinstance(var, Orange.feature.Continuous)


def is_multitarget(domain):
    return bool(domain.class_vars)


def is_multitarget_discrete(domain):
    return all(map(is_discrete, domain.class_vars))


def is_multitarget_continuous(domain):
    return all(map(is_continuous, domain.class_vars))


class OWTestMultitargetLearners(OWTestLearners):
    contextHandlers = {}

    cStatistics = \
        [Score(*s) for s in
         [("Average Logloss", "Logloss (average)", avg_logloss, True),
          ("Flattened Logloss", "Logloss (flattened)", flat_logloss, False),
          ("Global Accuracy", "Global Accuracy",
           Orange.multitarget.scoring.mt_global_accuracy, True),
          ("Mean Accuracy", "Mean Accuracy",
           Orange.multitarget.scoring.mt_mean_accuracy, True),
          ("Average Information Score", "Inf. Score (average)",
           avg_is, True),
          ("Flattened Information Score", "Inf. Score (flattened)",
           flat_bs, False),
          ("Average Brier Score", "Brier (average)", avg_bs, True),
          ("Flattened Brier Score", "Brier (flattened)", flat_bs, False),
          ("F1 macro", "F1 macro",
           Orange.evaluation.scoring.mlc_F1_macro, False),
          ("F1 micro", "F1 micro",
           Orange.evaluation.scoring.mlc_F1_micro, False)]
         ]

    # Regression
    rStatistics = \
        [Score(*s) for s in
         [("Average RMSE", "RMSE (average)", avg_rmse, True),
          ("Flattened RMSE", "RMSE (flattened)", flat_rmse, True)]
         ]

    def __init__(self, parent=None, signalManager=None,
                 title="Test Multitarget Learners"):
        OWTestLearners.__init__(self, parent, signalManager)
        self.setCaption(title)

        self.inputs = [("Data", Orange.data.Table, self.setData, Default),
                       ("Separate Test Data", Orange.data.Table,
                        self.setTestData),
                       ("Learner", Orange.core.Learner, self.setLearner,
                        Default + Multiple),
                       ("Preprocess", PreprocessedLearner,
                        self.setPreprocessor)]

        self.outputs = [("Evaluation Results",
                         Orange.evaluation.testing.ExperimentResults)]

        # Hide the "Target class" group box
        for box in self.controlArea.findChildren(QGroupBox):
            if str(box.title()).strip() == "Target class":
                box.hide()

    def invalidate(self, learner_id):
        """Invalidate results and scores for learner_id
        """
        self.learners[learner_id].scores = []
        self.learners[learner_id].results = None

    def invalidateAll(self):
        for learner_id in self.learners:
            self.invalidate(learner_id)

    def removeLearner(self, learner_id):
        """Remove the learner.
        """
        # Remove the results for this learner (if shared).
        res = self.learners[learner_id].results
        if res and res.number_of_learners > 1:
            old_learner = self.learners[learner_id].learner
            indx = res.learners.index(old_learner)
            res.remove(indx)
            del res.learners[indx]
        del self.learners[learner_id]

    def setData(self, data=None):
        self.error([0, 1])
        if data is not None and not is_multitarget(data.domain):
            data = None
            self.error(0, "Multitarget data expected.")

        if data is None:
            self.data = None
            self.clearScores()
            self.send("Evaluation Results", None)
            return

        self.clearScores()
        if is_multitarget_discrete(data.domain):
            self.statLayout.setCurrentWidget(self.cbox)
            self.stat = self.cStatistics
        elif is_multitarget_continuous(data.domain):
            self.statLayout.setCurrentWidget(self.rbox)
            self.stat = self.rStatistics
        elif is_multitarget(data.domain):
            self.error(1, "Mixed targets not supported")
            data = None
        else:
            self.warning(1, "Multi target domain expected.")

        self.data = data

        self.invalidateAll()

    def setTestData(self, data=None):
        self.testdata = data
        self.testDataBtn.setEnabled(data is not None)
        if self.resampling == 4:
            # Invalidate all scores.
            self.invalidateAll()

    def setLearner(self, learner=None, id=None):
        if learner is not None:
            if id in self.learners:
                self.invalidate(id)
                self.learners[id].learner = learner
                self.learners[id].name = learner.name
            else:
                self.learners[id] = Learner(learner, id)
        elif id in self.learners:
            self.removeLearner(id)

    def setPreprocessor(self, pp=None):
        self.preprocessor = pp

        self.invalidateAll()

    def handleNewSignals(self):
        self.updateScores()

    def updateScores(self):
        """Update the results/scores that are in need of updating.
        """
        def needsupdate(learner_id):
            return not (self.learners[learner_id].scores or \
                        self.learners[learner_id].results)

        self.score(filter(needsupdate, self.learners))
        self.paintscores()

    def score(self, learner_ids):
        """Compute scores for the list of learner ids.
        """
        if not self.data:
            return
        learners = []
        used_ids = []
        n = len(self.data.domain.attributes) * 2
        indices = sample.SubsetIndices2(
                    p0=min(n, len(self.data)),
                    stratified=sample.SubsetIndices2.StratifiedIfPossible)
        new = self.data.selectref(indices(self.data))

        self.warning(0)
        learner_exceptions = []

        for learner_id in learner_ids:
            learner = self.learners[learner_id].learner
            if self.preprocessor:
                learner = self.preprocessor.wrapLearner(learner)
            try:
                predictor = learner(new)
                predicted = predictor(new[0])

                if all(v.varType == c.varType for v, c in \
                       zip(predicted, new.domain.class_vars)):
                    learners.append(learner)
                    used_ids.append(learner_id)
                else:
                    self.learners[learner_id].scores = []
                    self.learners[learner_id].results = None

            except Exception, ex:
                learner_exceptions.append((self.learners[learner_id], ex))

        if learner_exceptions:
            text = "\n".join("Learner %r ends with exception: %r" % \
                             (learn.name, ex) for learn, ex in \
                             learner_exceptions)
            self.warning(0, text)

        if not learners:
            return

        # computation of results
        pb = None
        if self.resampling == 0:
            # Cross validation
            pb = OWGUI.ProgressBar(self, iterations=self.nFolds)
            res = testing.cross_validation(
                    learners, self.data, folds=self.nFolds,
                    strat=sample.SubsetIndices.StratifiedIfPossible,
                    callback=pb.advance,
                    store_examples=True)

            pb.finish()
        elif self.resampling == 1:
            # Leave one out
            pb = OWGUI.ProgressBar(self, iterations=len(self.data))
            res = testing.leave_one_out(
                    learners, self.data,
                    callback=pb.advance,
                    store_examples=True)

            pb.finish()
        elif self.resampling == 2:
            pb = OWGUI.ProgressBar(self, iterations=self.pRepeat)
            res = testing.proportion_test(
                    learners, self.data,
                    self.pLearning / 100.,
                    times=self.pRepeat,
                    callback=pb.advance,
                    store_examples=True)

            pb.finish()
        elif self.resampling == 3:
            pb = OWGUI.ProgressBar(self, iterations=len(learners))
            res = testing.learn_and_test_on_learn_data(
                    learners, self.data,
                    store_examples=True,
                    callback=pb.advance)

            pb.finish()

        elif self.resampling == 4:
            if not self.testdata:
                for l in self.learners.values():
                    l.scores = []
                return
            pb = OWGUI.ProgressBar(self, iterations=len(learners))
            res = testing.learn_and_test_on_test_data(
                    learners, self.data, self.testdata,
                    store_examples=True,
                    callback=pb.advance)

            pb.finish()

        if self.preprocessor:
            # Unwrap learners
            learners = [l.wrappedLearner for l in learners]

        res.learners = learners

        for lid in learner_ids:
            learner = self.learners[lid]
            if learner.learner in learners:
                learner.results = res
            else:
                learner.results = None

        self.error(range(len(self.stat)))
        scores = []

        for i, s in enumerate(self.stat):
            if s.cmBased:
                try:
                    scores.append(s.f(res))
                except Exception, ex:
                    self.error(
                        i, "An error occurred while evaluating " + \
                        str(s.f) + "on %s due to %s" % \
                        (" ".join([l.name for l in learners]), ex.message))

                    scores.append([None] * len(self.learners))
            else:
                scores_one = []
                for res_one in scoring.split_by_classifiers(res):
                    try:
                        scores_one.extend(s.f(res_one))
                    except Exception, ex:
                        self.error(
                            i, "An error occurred while evaluating " +\
                            str(s.f) + "on %s due to %s" % \
                            (res.classifierNames[0], ex.message))

                        scores_one.append(None)
                        import traceback
                        traceback.print_exc()
                scores.append(scores_one)

        for i, (lid, l) in enumerate(zip(used_ids, learners)):
            self.learners[lid].scores = [s[i] if s else None for s in scores]

        self.sendResults()

    def get_usestat(self):
        stats = [self.selectedCScores, self.selectedRScores]
        if self.data is None:
            return stats[self.statLayout.currentIndex()]
        if is_multitarget_continuous(self.data.domain):
            return self.selectedRScores
        elif is_multitarget_discrete(self.data.domain):
            return self.selectedCScores
        else:
            raise ValueError()

    def sendReport(self):
        exset = []
        if self.resampling == 0:
            exset = [("Folds", self.nFolds)]
        elif self.resampling == 2:
            exset = [("Repetitions", self.pRepeat),
                     ("Proportion of training instances",
                      "%i%%" % self.pLearning)]
        else:
            exset = []

        self.reportSettings(
            "Validation method",
            [("Method", self.resamplingMethods[self.resampling])] + exset)

        self.reportData(self.data)

        if self.data:
            self.reportSection("Results")
            learners = [(l.time, l) for l in self.learners.values()]
            learners.sort()
            learners = [lt[1] for lt in learners]
            usestat = self.get_usestat()
            res = "<table><tr><th></th>" + \
                  "".join("<th><b>%s</b></th>" % hr for hr in \
                          [s.label for i, s in enumerate(self.stat)
                           if i in usestat]) + \
                  "</tr>"
            for i, l in enumerate(learners):
                res += "<tr><th><b>%s</b></th>" % l.name
                if l.scores:
                    for j in usestat:
                        scr = l.scores[j]
                        res += "<td>" + \
                               ("%.4f" % scr if scr is not None else "") + \
                               "</td>"
                res += "</tr>"
            res += "</table>"
            self.reportRaw(res)


if __name__ == "__main__":
    a = QApplication(sys.argv[1:])
    ow = OWTestMultitargetLearners()

    data1 = Orange.data.Table("multitarget:bridges.tab")
    data2 = Orange.data.Table("multitarget:emotions.tab")
    data3 = Orange.data.Table("multitarget-synthetic.tab")

    l1 = Orange.classification.majority.MajorityLearner(name="Majority")
    l2 = Orange.classification.bayes.NaiveLearner(name="Naive Bayes")

    l1 = Orange.multitarget.binary.BinaryRelevanceLearner(learner=l1,
                                                          name=l1.name)
    l2 = Orange.multitarget.binary.BinaryRelevanceLearner(learner=l2,
                                                          name=l2.name)

    ow.setData(data1)
    ow.setLearner(l1, 1)
    ow.setLearner(l2, 2)
    ow.handleNewSignals()

    ow.show()
    a.exec_()

    ow.setData(data2)
    ow.setTestData(data2)
#    ow.handleNewSignals()

#    ow.show()
#    a.exec_()

    l3 = Orange.regression.earth.EarthLearner(name="Earth")
    l4 = Orange.regression.pls.PLSRegressionLearner(name="PLS")

    ow.setLearner(None, 1)
    ow.setLearner(None, 2)

    ow.setLearner(l3, 3)
    ow.setLearner(l4, 4)
    ow.setData(data3)

    ow.handleNewSignals()

    ow.show()
    a.exec_()
#    ow.saveSettings()
