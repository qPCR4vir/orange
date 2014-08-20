"""
<name>Neural Network</name>
<description>Neural network learner/classifier supporting multi target problems.</description>
<category>Multitarget</category>
<priority>20<priority>
<tags>neural,network,multitarget</tags>
<icon>icons/NeuralNetwork.png</icon>

"""

import Orange
import Orange.multitarget
from orngWrap import PreprocessedLearner

from OWWidget import *
import OWGUI

class OWNeuralNetwork(OWWidget):
    settingsList = ["name", "n_mid", "reg_fact", "max_iter"]

    def __init__(self, parent=None, signalManager=None,
                 title="Neural Network"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data),
                       ("Preprocess", PreprocessedLearner,
                        self.set_preprocessor)
                       ]
        self.outputs = [("Learner", Orange.classification.Learner),
                        ("Classifier", Orange.classification.Classifier)
                        ]

        self.name = "Neural Network"
        self.n_mid = 20
        self.reg_fact = 1
        self.max_iter = 1000

        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Name", addSpace=True)
        OWGUI.lineEdit(box, self, "name")

        box = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        OWGUI.spin(box, self, "n_mid", 2, 10000, 1,
                   label="Hidden layer neurons",
                   tooltip="Number of neurons in the hidden layer."
                   )

        OWGUI.doubleSpin(box, self, "reg_fact", 0.1, 10.0, 0.1,
                         label="Regularization factor",
                         )

        OWGUI.spin(box, self, "max_iter", 100, 10000, 1,
                   label="Max iterations",
                   tooltip="Maximal number of optimization iterations."
                   )

        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     tooltip="Create the learner and apply it on input data.",
                     autoDefault=True
                     )

        self.data = None
        self.preprocessor = None
        self.apply()

    def set_data(self, data=None):
        self.data = data
        self.error([0])
        if data is not None and not data.domain.class_vars:
            data = None
            self.error(0, "Input data must have multi target domain.")

        self.data = data
        self.apply()

    def set_preprocessor(self, preprocessor=None):
        self.preprocessor = preprocessor

    def apply(self):
        learner = Orange.multitarget.neural.NeuralNetworkLearner(
            name=self.name, n_mid=self.n_mid,
            reg_fact=self.reg_fact, max_iter=self.max_iter
        )

        if self.preprocessor is not None:
            learner = self.preprocessor.wrapLearner(learner)

        classifier = None
        self.error([1])
        if self.data is not None:
            try:
                classifier = learner(self.data)
                classifier.name = self.name
            except Exception, ex:
                self.error(1, str(ex))

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def sendReport(self):
        self.reportSettings("Parameters",
                            [("Hidden layer neurons", self.n_mid),
                             ("Regularization factor", self.reg_fact),
                             ("Max iterations", self.max_iter)]
                            )


if __name__ == "__main__":
    app = QApplication([])
    w = OWNeuralNetwork()
    data = Orange.data.Table("multitarget:emotions.tab")
    w.set_data(data)
    w.set_data(None)
    w.set_data(data)
    w.show()
    app.exec_()
