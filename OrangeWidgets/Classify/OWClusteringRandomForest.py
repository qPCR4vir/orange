"""
<name>Clustering Random Forest</name>
<description>Clustering random forest learner/classifier for multi
target classification.</description>
<icon>icons/ClusteringRandomForest.png</icon>
<priority>110</priority>
"""

import Orange
import Orange.multitarget

from Orange.tuning import PreprocessedLearner

from OWWidget import *
import OWGUI

class OWClusteringRandomForest(OWWidget):
    settingsList = ["name", "trees", "min_instances", "min_majority",
                    "max_depth", "min_MSE", "method"]

    def __init__(self, parent=None, signalManager=None,
                 title="Clustering Random Forest"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False)

        self.inputs = [("Data", Orange.data.Table, self.set_data),
                       ("Preprocess", PreprocessedLearner,
                        self.set_preprocessor)
                       ]
        self.outputs = [("Learner", Orange.classification.Learner),
                        ("Classifier", Orange.classification.Classifier)
                        ]

        self.name = 'Clustering Random Forest'
        self.trees = 25
        self.max_depth = 100
        self.min_majority = 1.0
        self.min_MSE = 0.001
        self.min_instances = 5
        self.method = 0

        self.maxTrees = 10000

        self.loadSettings()

        self.data = None
        self.preprocessor = None

        box = OWGUI.widgetBox(self.controlArea, "Classifier/Learner Name")
        OWGUI.lineEdit(box, self, "name")

        OWGUI.spin(box, self, "trees", 1, 1000, orientation="horizontal", label="Number of trees in forest")

        OWGUI.separator(self.controlArea)

        OWGUI.spin(box, self, "max_depth", 1, 1000, 1,
                   "Stop splitting nodes at depth")

        OWGUI.doubleSpin(box, self, "min_majority", 0.01, 1.0, 0.01,
                         "Minimal majority class proportion (%)",
                         tooltip="Minimal proportion of the majority class "
                                 "value each of the class variables has to "
                                 "reach to stop induction (only used for "
                                 "classification)"
                        )

        OWGUI.doubleSpin(box, self, "min_MSE", 0.001, 1.0, 0.001,
                         "Min. mean squared error",
                         tooltip="Minimal mean squared error each of the "
                                 "class variables has to reach to stop "
                                 "induction (only used for regression)."
                         )

        OWGUI.spin(box, self, "min_instances", 1, 1000, 1,
                   "Min. instances in leaves")


        OWGUI.radioButtonsInBox(self.controlArea, self, "method",
              box = "Feature scorer",
              btnLabels = ["Inter dist", "Intra dist", "Silhouette" ,"Gini-index"],
              tooltips = ["Maximal distance between clusters",
                          "Minimal distance inside clusters ",
                          "Silhouette measure with prototypes",
                          "Gini-index, used for nominal class variables"]
                          )

        OWGUI.separator(self.controlArea)

        OWGUI.button(self.controlArea, self, "&Apply",
                     callback=self.apply,
                     tooltip="Create the learner and apply it on input data.",
                     autoDefault=True
                     )

        self.data = None
        self.preprocessor = None
        self.apply()

    def sendReport(self):
        self.reportSettings(
            "Parameters",
            [("# of trees", self.trees),
            ("Max depth", self.max_depth),
             ("Min. majority", self.min_majority),
             ("Min MSE", self.min_MSE),
             ("Min instances in leaves", self.min_instances)
             ]
        )
        self.reportData(self.data)


    def set_data(self, data=None):
        """Set the widget input data.
        """
        self.data = data
        self.error([0])
        if data is not None and not data.domain.class_vars:
            data = None
            self.error(0, "Input data must have multi target domain.")

        self.data = data
        self.apply()

        self.send("Random Forest Classifier", self.classifier)
        
    def set_preprocessor(self, preprocessor=None):
        """Set data preprocessor.
        """
        self.preprocessor = preprocessor



    def apply(self):
        """Apply the settings to the output learner. If input data is available
        then also construct a classifier.

        """

        learner = Orange.multitarget.tree.ClusteringTreeLearner(
                    max_depth=self.max_depth,
                    min_majority=self.min_majority,
                    min_MSE=self.min_MSE,
                    min_instances=self.min_instances,
                    method=self.method,
                    name=self.name)

        learner = Orange.ensemble.forest.RandomForestLearner(base_learner=learner, \
            trees=self.trees, name=self.name)

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




if __name__=="__main__":
    # no tests currently
    pass
