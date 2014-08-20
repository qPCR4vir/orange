"""<name>PLS Classification</name>
<description>PLS Classification</description>
<category>Multitarget</category>
<priority>160</priority>
<icon>icons/PLSClassification.png</icon>
"""

import Orange
import Orange.multitarget

from OWWidget import *
import OWGUI

from orngWrap import PreprocessedLearner


class OWPLSClassifier(OWWidget):
    settings = ["name", "n_comp"]

    def __init__(self, parent=None, signalManager=None,
                 title="PLS Classifier"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data),
                       ("Preprocess", PreprocessedLearner,
                        self.set_preprocessor)
                       ]
        self.outputs = [("Learner", Orange.classification.Learner),
                        ("Classifier", Orange.classification.Classifier)
                        ]

        self.name = "PLS Classification Learner"
        self.n_comp = 2

        box = OWGUI.widgetBox(self.controlArea, box="Name", addSpace=True)
        OWGUI.lineEdit(box, self, "name")

        OWGUI.spin(self.controlArea, self, "n_comp", 1, 100, 1,
                   "Num. of components to keep")

        OWGUI.button(self.controlArea, self, "Apply",
                     callback=self.apply,
                     default=True
                     )

        self.data = None
        self.preprocessor = None
        self.apply()

    def set_data(self, data=None):
        """Set the input data.
        """
        self.error(0)
        if data is not None and not data.domain.class_vars:
            self.error(0, "Data with multi-target class expected.")
            data = None

        self.data = data

    def set_preprocessor(self, preprocessor=None):
        """Set data preprocessor.
        """
        self.preprocessor = preprocessor

    def handleNewSignals(self):
        self.apply()

    def apply(self):
        """
        """
        learner = Orange.multitarget.pls.PLSClassificationLearner(
                    n_comp=self.n_comp,
                    name=self.name
                    )

        if self.preprocessor is not None:
            learner = self.preprocessor.wrapLearner(learner)

        classifier = None
        self.error(1)
        if self.data is not None:
            try:
                classifier = learner(self.data)
                classifier.name = self.name
            except Exception, ex:
                self.error(1, str(ex))

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def sendReport(self):
        self.reportSettings(
            "Parameters",
            [("Number of components to keep", self.n_comp)]
        )

        self.reportData(self.data)
