"""
<name>Ensemble Classifier Chain</name>
<description>Train an ensemble of chain classifiers</description>
<priority>1200</priority>
<category>Multitarget</category>
<tags>wrapper,multitarget,chain,ensemble</tags>
<icon>icons/EnsembleClassifierChain.png</icon>
"""

import Orange
import Orange.multitarget

from OWWidget import *
import OWGUI


class OWEnsembleClassifierChain(OWWidget):
    settingsList = ["name", "n_chains", "sample_size", "actual_values"]

    def __init__(self, parent=None, signalManager=None,
                 title="Ensemble Classifier Chain"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data),
                       ("Base Learner", Orange.classification.Learner,
                        self.set_learner)]

        self.outputs = [("Learner", Orange.classification.Learner),
                        ("Classifier", Orange.classification.Classifier)]

        self.name = "Ensemble Classifier Chain"
        self.n_chains = 50
        self.sample_size = 0.25
        self.actual_values = False

        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Learner/Classifier Name",
                              addSpace=True)
        OWGUI.lineEdit(box, self, "name")

        box = OWGUI.widgetBox(self.controlArea, "Settings",
                              addSpace=True)

        OWGUI.spin(box, self, "n_chains", 5, 1000,
                   label="Num. of chains",
                   tooltip="Number of chains to be constructed.")

        OWGUI.doubleSpin(box, self, "sample_size", 0.05, 1.0, 0.05,
                         label="Sample size",
                         tooltip="Build each classifier on a fraction of the "
                                 "original data."
                        )

        OWGUI.checkBox(box, self, "actual_values", 
                   label="Use actual values",
                   tooltip="Use actual values insteand of predicted ones when adding classes into features.")

        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     autoDefault=True)

        self.base_learner = None
        self.data = None
        self.apply()

    def set_data(self, data=None):
        self.error([0])
        if data is not None and not data.domain.class_vars:
            data = None
            self.error(0, "Input data must have multi target domain.")

        self.data = data

    def set_learner(self, base_learner=None):
        self.base_learner = base_learner

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        learner = None
        if self.base_learner is not None:
            learner = Orange.multitarget.chain.EnsembleClassifierChainLearner(
                name=self.name, learner=self.base_learner,
                n_chains=self.n_chains, sample_size=self.sample_size, actual_values=self.actual_values
            )

        classifier = None
        if self.data is not None and learner is not None:
            classifier = learner(self.data)
            classifier.name = self.name

        self.send("Learner", learner)
        self.send("Classifier", classifier)


if __name__ == "__main__":
    app = QApplication([])
    w = OWEnsembleClassifierChain()
    data = Orange.data.Table("multitarget:emotions.tab")
    base_learner = Orange.classification.bayes.NaiveLearner()
    w.set_data(data)
    w.set_learner(base_learner)
    w.set_data(None)
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()
    w.saveSettings()
