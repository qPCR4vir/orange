"""
<name>Binary Relevance</name>
<description>Binary relevance learner wrapper</description>
<priority>1000</priority>
<category>Multitarget</category>
<tags>multitarget,binary,relevance,wrapper</tags>
<icon>icons/BinaryRelevance.png</icon>

"""

import Orange
import Orange.multitarget

from OWWidget import *
import OWGUI


class OWBinaryRelevance(OWWidget):
    settingsList = ["name"]

    def __init__(self, parent=None, signalManager=None,
                 title="Binary Relevance"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data),
                       ("Base Learner", Orange.classification.Learner,
                        self.set_learner)]

        self.outputs = [("Learner", Orange.classification.Learner),
                        ("Classifier", Orange.classification.Classifier)]

        self.name = "Binary Relevance"

        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Learner/Classifier Name")
        OWGUI.lineEdit(box, self, "name")

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
            learner = Orange.multitarget.binary.BinaryRelevanceLearner(
                name=self.name, learner=self.base_learner
            )

        classifier = None
        if self.data is not None and learner is not None:
            classifier = learner(self.data)
            classifier.name = self.name

        self.send("Learner", learner)
        self.send("Classifier", classifier)


if __name__ == "__main__":
    app = QApplication([])
    w = OWBinaryRelevance()
    data = Orange.data.Table("multitarget:emotions.tab")
    base_learner = Orange.classification.bayes.NaiveLearner()
    w.set_data(data)
    w.set_learner(base_learner)
    w.set_data(None)
    w.set_data(data)
    w.show()
    app.exec_()
    w.saveSettings()
