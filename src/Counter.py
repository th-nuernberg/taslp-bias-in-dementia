"""
@author: Kashaf Gulzar

computes metrics summary for acoustic_classification.py
"""
import numpy as np

class ClassifierCounter:
    def __init__(self):
        self.tn_hc_age_groups_total = [0, 0]
        self.tn_hc_genders_total = [0, 0]
        self.tn_hc_mmse_total = [0, 0, 0]
        self.tn_hc_ad_status_total = [0, 0]
        self.tn_hc_depression_status_total = [0, 0]

        self.fp_hc_age_groups_total = [0, 0]
        self.fp_hc_genders_total = [0, 0]
        self.fp_hc_mmse_total = [0, 0, 0]
        self.fp_hc_ad_status_total = [0, 0]
        self.fp_hc_depression_status_total = [0, 0]

        self.tp_ad_age_groups_total = [0, 0]
        self.tp_ad_genders_total = [0, 0]
        self.tp_ad_mmse_total = [0, 0, 0]
        self.tp_ad_ad_status_total = [0, 0]
        self.tp_ad_depression_status_total = [0, 0]

        self.fn_ad_age_groups_total = [0, 0]
        self.fn_ad_genders_total = [0, 0]
        self.fn_ad_mmse_total = [0, 0, 0]
        self.fn_ad_ad_status_total = [0, 0]
        self.fn_ad_depression_status_total = [0, 0]

    def update_counts(self,tn_age,tn_genders,tn_mmse,tn_ad,tn_dep,fp_age,fp_genders,fp_mmse,fp_ad,fp_dep,tp_age,tp_genders,tp_mmse,tp_ad,tp_dep,fn_age,fn_genders,fn_mmse,fn_ad,fn_dep):
        self.tn_hc_age_groups_total = np.add(self.tn_hc_age_groups_total, tn_age)
        self.tn_hc_genders_total = np.add(self.tn_hc_genders_total, tn_genders)
        self.tn_hc_mmse_total = np.add(self.tn_hc_mmse_total, tn_mmse)
        self.tn_hc_ad_status_total = np.add(self.tn_hc_ad_status_total, tn_ad)
        self.tn_hc_depression_status_total = np.add(self.tn_hc_depression_status_total, tn_dep)

        self.fp_hc_age_groups_total = np.add(self.fp_hc_age_groups_total, fp_age)
        self.fp_hc_genders_total = np.add(self.fp_hc_genders_total, fp_genders)
        self.fp_hc_mmse_total = np.add(self.fp_hc_mmse_total, fp_mmse)
        self.fp_hc_ad_status_total = np.add(self.fp_hc_ad_status_total, fp_ad)
        self.fp_hc_depression_status_total = np.add(self.fp_hc_depression_status_total, fp_dep)

        self.tp_ad_age_groups_total = np.add(self.tp_ad_age_groups_total, tp_age)
        self.tp_ad_genders_total = np.add(self.tp_ad_genders_total, tp_genders)
        self.tp_ad_mmse_total = np.add(self.tp_ad_mmse_total, tp_mmse)
        self.tp_ad_ad_status_total = np.add(self.tp_ad_ad_status_total, tp_ad)
        self.tp_ad_depression_status_total = np.add(self.tp_ad_depression_status_total, tp_dep)

        self.fn_ad_age_groups_total = np.add(self.fn_ad_age_groups_total, fn_age)
        self.fn_ad_genders_total = np.add(self.fn_ad_genders_total, fn_genders)
        self.fn_ad_mmse_total = np.add(self.fn_ad_mmse_total, fn_mmse)
        self.fn_ad_ad_status_total = np.add(self.fn_ad_ad_status_total, fn_ad)
        self.fn_ad_depression_status_total = np.add(self.fn_ad_depression_status_total, fn_dep)

    def get_total_counts(self):

        return self.tn_hc_age_groups_total, self.tn_hc_genders_total, self.tn_hc_mmse_total, self.tn_hc_ad_status_total, \
               self.tn_hc_depression_status_total, self.fp_hc_age_groups_total, self.fp_hc_genders_total, self.fp_hc_mmse_total,self.fp_hc_ad_status_total,\
               self.fp_hc_depression_status_total, self.tp_ad_age_groups_total, self.tp_ad_genders_total, self.tp_ad_mmse_total,self.tp_ad_ad_status_total,\
               self.tp_ad_depression_status_total, self.fn_ad_age_groups_total, self.fn_ad_genders_total, self.fn_ad_mmse_total,self.fn_ad_ad_status_total,\
               self.fn_ad_depression_status_total


class Y_testCounter:

    def __init__(self):
        self.y_test_hc_age_groups_total = [0, 0]
        self.y_test_hc_genders_total = [0, 0]
        self.y_test_hc_mmse_total = [0, 0, 0]
        self.y_test_hc_ad_status_total = [0, 0]
        self.y_test_hc_depression_status_total = [0, 0]

        self.y_test_ad_age_groups_total = [0, 0]
        self.y_test_ad_genders_total = [0, 0]
        self.y_test_ad_mmse_total = [0, 0, 0]
        self.y_test_ad_ad_status_total = [0, 0]
        self.y_test_ad_depression_status_total = [0, 0]

    def update_counts(self, hc_age, hc_genders, hc_mmse, hc_ad_status, hc_dep, ad_age, ad_genders, ad_mmse, ad_ad_status, ad_dep):
        self.y_test_hc_age_groups_total = np.add(self.y_test_hc_age_groups_total, hc_age)
        self.y_test_hc_genders_total = np.add(self.y_test_hc_genders_total, hc_genders)
        self.y_test_hc_mmse_total = np.add(self.y_test_hc_mmse_total, hc_mmse)
        self.y_test_hc_ad_status_total = np.add(self.y_test_hc_ad_status_total, hc_ad_status)
        self.y_test_hc_depression_status_total = np.add(self.y_test_hc_depression_status_total, hc_dep)

        self.y_test_ad_age_groups_total = np.add(self.y_test_ad_age_groups_total, ad_age)
        self.y_test_ad_genders_total = np.add(self.y_test_ad_genders_total, ad_genders)
        self.y_test_ad_mmse_total = np.add(self.y_test_ad_mmse_total, ad_mmse)
        self.y_test_ad_ad_status_total = np.add(self.y_test_ad_ad_status_total, ad_ad_status)
        self.y_test_ad_depression_status_total = np.add(self.y_test_ad_depression_status_total, ad_dep)

    def get_total_counts(self):

        return self.y_test_hc_age_groups_total, self.y_test_hc_genders_total, self.y_test_hc_mmse_total, self.y_test_hc_ad_status_total,\
               self.y_test_hc_depression_status_total, self.y_test_ad_age_groups_total, self.y_test_ad_genders_total, \
               self.y_test_ad_mmse_total, self.y_test_ad_ad_status_total, self.y_test_ad_depression_status_total
