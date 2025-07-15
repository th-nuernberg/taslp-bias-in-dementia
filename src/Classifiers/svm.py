"""
@author: Kashaf Gulzar

classifier file imported in acoustic_classification.py
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from ..metrics import compute_metrics
import numpy as np
import seaborn as sns
import pickle


class SVM:

    def __init__(self, x_train, y_train, x_test, y_test, save_path_opt, save_model_path):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = None
        self.save_path_opt = save_path_opt
        self.model_path = save_model_path

    def svm_opt(self):
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}

        model = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
        # fitting the model for grid search
        model.fit(self.x_train, self.y_train)
        # save the model to disk
        pickle.dump(model, open(self.model_path, 'wb'))
        # load the model from disk
        loaded_model = pickle.load(open(self.model_path, 'rb'))
        # scores = loaded_model.decision_function(self.x_test)
        # print(scores)
        # Predict the response for test dataset
        self.y_pred = loaded_model.predict(self.x_test)
        # display and save confusion matrix
        cm = metrics.confusion_matrix(self.y_test, self.y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_metrics = compute_metrics(cm)
        print(f"Metrics of SVM Optimized: {cm_metrics}")
        plt.figure()
        ax = sns.heatmap(cm_norm, annot=True, fmt='.0%', vmin=0, vmax=1)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_xlabel("Predicted Diagnosis", fontsize=12, labelpad=6)
        ax.xaxis.set_ticklabels(['Non-depressed AD', 'Depressed AD'])
        ax.set_ylabel("Actual Diagnosis", fontsize=12, labelpad=6)
        ax.yaxis.set_ticklabels(['Non-depressed AD', 'Depressed AD'])
        ax.set_title("SVM Result", fontsize=12)
        plt.savefig(self.save_path_opt)
        #plt.show()
        return cm_metrics

    def get_error_idx(self):
        tn_hc_indices = []
        fp_hc_indices = []
        tp_ad_indices = []
        fn_ad_indices = []
        y_test_idx = self.y_test.index
        y_test_list = list(self.y_test)
        y_pred_list = list(self.y_pred)
        for i in range(len(y_test_list)):
            if y_test_list[i] == 0 and y_pred_list[i] == 0:
                tn_hc_indices.append(y_test_idx[i])
            elif y_test_list[i] == 0 and y_pred_list[i] == 1:
                fp_hc_indices.append(y_test_idx[i])
            elif y_test_list[i] == 1 and y_pred_list[i] == 1:
                tp_ad_indices.append(y_test_idx[i])
            elif y_test_list[i] == 1 and y_pred_list[i] == 0:
                fn_ad_indices.append(y_test_idx[i])
        return tn_hc_indices, fp_hc_indices, tp_ad_indices,fn_ad_indices
