"""
@author: Kashaf Gulzar

Computes subgroup-wise performance metrics and populates them to the table
"""

# -------------- Import Modules ---------------
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------- Biasness check ---------------
class BiasnessData:

    def __init__(self, original_df, indices):
        self.df = original_df
        self.indices = indices
        self.filenames = None
        self.ages = None
        self.genders = None
        self.mmse_score = None
        self.ad_status = None
        self.depression_status = None

    def get_metadata(self):
        self.filenames = self.df.Filename.loc[self.indices]
        self.ages = list(self.df.Age.loc[self.indices])
        self.genders = list(self.df.Gender.loc[self.indices])
        self.mmse_score = list(self.df.MMSE_score.loc[self.indices])
        self.ad_status = list(self.df.Ad_status.loc[self.indices])
        self.depression_status = list(self.df.Depression_status.loc[self.indices])

    def get_counts(self):
        age_group_1 = 0
        age_group_2 = 0
        males = 0
        females = 0
        ad_severe = 0
        ad_mild = 0
        ad_low = 0
        hc_ad_status = 0
        ad_ad_status = 0
        hc_dep_status = 0
        dep_dep_status = 0

        for i in range(len(self.ages)):
            if 0 <= self.ages[i] <= 65:
                age_group_1 += 1
            elif 66 <= self.ages[i] <= 100:
                age_group_2 += 1
        age_groups = [age_group_1, age_group_2]

        for i in range(len(self.genders)):
            if self.genders[i] == 0:
                females += 1
            elif self.genders[i] == 1:
                males += 1
        genders = [males, females]

        for i in range(len(self.mmse_score)):
            if 0 <= self.mmse_score[i] <= 15:
                ad_severe += 1
            elif 16 <= self.mmse_score[i] <= 23:
                ad_mild += 1
            elif 24 <= self.mmse_score[i] <= 30:
                ad_low += 1
        mmse = [ad_low, ad_mild, ad_severe]

        for i in range(len(self.ad_status)):
            if self.ad_status[i] == 0:
                hc_ad_status += 1
            elif self.ad_status[i] == 1:
                ad_ad_status += 1
        ad_status = [hc_ad_status, ad_ad_status]

        for i in range(len(self.depression_status)):
            if self.depression_status[i] == 0:
                hc_dep_status += 1
            elif self.depression_status[i] == 1:
                dep_dep_status += 1
        depression_status = [hc_dep_status, dep_dep_status]
        return age_groups, genders, mmse, ad_status, depression_status


class BiasnessTable:

    def __init__(self, age_groups: list, genders: list, mmse: list, ad_status: list, depression_status: list,
                 y_test_age_groups: list, y_test_genders: list, y_test_mmse: list, y_test_ad_status: list, y_test_depression_status: list):
        self.age_groups = age_groups
        self.genders = genders
        self.mmse = mmse
        self.ad_status = ad_status
        self.depression_status = depression_status
        self.y_test_age_groups = y_test_age_groups
        self.y_test_genders = y_test_genders
        self.y_test_mmse = y_test_mmse
        self.y_test_ad_status = y_test_ad_status
        self.y_test_depression_status = y_test_depression_status

    def get_table_data(self):
        # age groups relative to test set in percentage
        age_groups_perc = [0, 0]
        age_groups_perc[0] = (self.age_groups[0] / self.y_test_age_groups[0]) * 100
        age_groups_perc[1] = (self.age_groups[1] / self.y_test_age_groups[1]) * 100

        # genders relative to test set in percentage
        genders_perc = [0, 0]
        genders_perc[0] = (self.genders[0] / self.y_test_genders[0]) * 100
        genders_perc[1] = (self.genders[1] / self.y_test_genders[1]) * 100

        # mmse relative to test set in percentage
        mmse_perc = [0, 0, 0]
        mmse_perc[0] = (self.mmse[0] / self.y_test_mmse[0]) * 100
        mmse_perc[1] = (self.mmse[1] / self.y_test_mmse[1]) * 100
        mmse_perc[2] = (self.mmse[2] / self.y_test_mmse[2]) * 100
        mmse_perc = [0 if math.isnan(x) else x for x in mmse_perc]

        # AD status relative to test set in percentage
        ad_status_perc = [0, 0]
        ad_status_perc[0] = (self.ad_status[0] / self.y_test_ad_status[0]) * 100
        ad_status_perc[1] = (self.ad_status[1] / self.y_test_ad_status[1]) * 100

        # Depression status relative to test set in percentage
        dep_status_perc = [0, 0]
        dep_status_perc[0] = (self.depression_status[0] / self.y_test_depression_status[0]) * 100
        dep_status_perc[1] = (self.depression_status[1] / self.y_test_depression_status[1]) * 100

        return age_groups_perc, genders_perc, mmse_perc, ad_status_perc, dep_status_perc


class BiasnessPlots:

    def __init__(self, age_groups: list, genders: list, mmse: list, ad_status: list, depression_status: list,
                 y_test_age_groups: list, y_test_genders: list, y_test_mmse: list, y_test_ad_status: list,
                 y_test_depression_status: list):
        self.age_groups = age_groups
        self.genders = genders
        self.mmse = mmse
        self.ad_status = ad_status
        self.depression_status = depression_status
        self.y_test_age_groups = y_test_age_groups
        self.y_test_genders = y_test_genders
        self.y_test_mmse = y_test_mmse
        self.y_test_ad_status = y_test_ad_status
        self.y_test_depression_status = y_test_depression_status

    def agegroup_plots(self, title: str, ylabel: str, save_path: str, test_plot_save_path: str):
        # age groups in %
        age_groups_perc = [0, 0]
        y_test_age_groups_perc = [0, 0]
        age_groups_perc[0] = (self.age_groups[0] / np.sum(self.age_groups)) * 100
        age_groups_perc[1] = (self.age_groups[1] / np.sum(self.age_groups)) * 100
        y_test_age_groups_perc[0] = (self.age_groups[0] / self.y_test_age_groups[0]) * 100
        y_test_age_groups_perc[1] = (self.age_groups[1] / self.y_test_age_groups[1]) * 100
        print(y_test_age_groups_perc)
        # plot age groups
        plt.figure(1)
        plt.bar(['Age group 1', 'Age group 2'], age_groups_perc)
        plt.xlabel('Age Groups', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plot age groups wrt test percentage
        plt.figure(2)
        plt.bar(['Age group 1', 'Age group 2'], y_test_age_groups_perc)
        plt.xlabel('Age Groups', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(test_plot_save_path)
        plt.show()
        return age_groups_perc, y_test_age_groups_perc

    def gender_plots(self, title: str, ylabel: str, save_path: str, test_plot_save_path: str):
        # genders in %
        genders_perc = [0, 0]
        y_test_genders_perc = [0, 0]
        genders_perc[0] = (self.genders[0] / np.sum(self.genders)) * 100
        genders_perc[1] = (self.genders[1] / np.sum(self.genders)) * 100
        print(genders_perc)
        y_test_genders_perc[0] = (self.genders[0] / self.y_test_genders[0]) * 100
        y_test_genders_perc[1] = (self.genders[1] / self.y_test_genders[1]) * 100
        print(y_test_genders_perc)
        # plot genders
        plt.figure(1)
        plt.bar(['Females', 'Males'], genders_perc)
        plt.xlabel('Gender', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plot genders wrt test percentage
        plt.figure(2)
        plt.bar(['Females', 'Males'], y_test_genders_perc)
        plt.xlabel('Gender', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(test_plot_save_path)
        plt.show()
        return genders_perc, y_test_genders_perc

    def mmse_plots(self, title: str, ylabel: str, save_path: str, test_plot_save_path: str):
        # mmse in %
        mmse_perc = [0, 0, 0]
        ytest_mmse_perc = [0, 0, 0]
        mmse_perc[0] = (self.mmse[0] / np.sum(self.mmse)) * 100
        mmse_perc[1] = (self.mmse[1] / np.sum(self.mmse)) * 100
        mmse_perc[2] = (self.mmse[2] / np.sum(self.mmse)) * 100
        print(mmse_perc)
        ytest_mmse_perc[0] = (self.mmse[0] / self.y_test_mmse[0]) * 100
        ytest_mmse_perc[1] = (self.mmse[1] / self.y_test_mmse[1]) * 100
        ytest_mmse_perc[2] = (self.mmse[2] / self.y_test_mmse[2]) * 100
        ytest_mmse_perc = [0 if math.isnan(x) else x for x in ytest_mmse_perc]
        print(ytest_mmse_perc)
        plt.figure(1)
        plt.bar(['Low', 'Mild', 'Severe'], mmse_perc)
        plt.xlabel('Severity of AD', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plot mmse wrt test percentage
        plt.figure(2)
        plt.bar(['Low', 'Mild', 'Severe'], ytest_mmse_perc)
        plt.xlabel('Severity of AD', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(test_plot_save_path)
        plt.show()
        return mmse_perc, ytest_mmse_perc

    def ad_status_plots(self, title: str, ylabel: str, save_path: str, test_plot_save_path: str):
        # ad status in %
        ad_status_perc = [0, 0]
        y_test_ad_status_perc = [0, 0]
        ad_status_perc[0] = (self.ad_status[0] / np.sum(self.ad_status)) * 100
        ad_status_perc[1] = (self.ad_status[1] / np.sum(self.ad_status)) * 100
        print(ad_status_perc)
        y_test_ad_status_perc[0] = (self.ad_status[0] / self.y_test_ad_status[0]) * 100
        y_test_ad_status_perc[1] = (self.ad_status[1] / self.y_test_ad_status[1]) * 100
        print(y_test_ad_status_perc)
        # plot as status
        plt.figure(1)
        plt.bar(['Healthy', 'AD Patient'], ad_status_perc)
        plt.xlabel('AD Status', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plot ad status wrt test percentage
        plt.figure(2)
        plt.bar(['Healthy', 'AD Patient'], y_test_ad_status_perc)
        plt.xlabel('AD Status', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(test_plot_save_path)
        plt.show()
        return ad_status_perc, y_test_ad_status_perc

    def depression_status_plots(self, title: str, ylabel: str, save_path: str, test_plot_save_path: str):
        # ad status in %
        dep_status_perc = [0, 0]
        y_test_dep_status_perc = [0, 0]
        dep_status_perc[0] = (self.depression_status[0] / np.sum(self.depression_status)) * 100
        dep_status_perc[1] = (self.depression_status[1] / np.sum(self.depression_status)) * 100
        print(dep_status_perc)
        y_test_dep_status_perc[0] = (self.depression_status[0] / self.y_test_depression_status[0]) * 100
        y_test_dep_status_perc[1] = (self.depression_status[1] / self.y_test_depression_status[1]) * 100
        print(y_test_dep_status_perc)
        # plot depression status
        plt.figure(1)
        plt.bar(['Healthy', 'Depressed'], dep_status_perc)
        plt.xlabel('Depression Status', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plot depression status wrt test percentage
        plt.figure(2)
        plt.bar(['Healthy', 'Depressed'], y_test_dep_status_perc)
        plt.xlabel('Depression Status', fontsize=10)
        plt.yticks(ticks=[0, 20, 40, 60, 80, 100], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(test_plot_save_path)
        plt.show()
        return dep_status_perc, y_test_dep_status_perc