import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_value_hists(axs, nonsurv_data, surv_data, step_num, var_idx=None, nb_bar=10):
    """
    Plot histograms of the computed values for Q_D, Q_R (for selected actions) and V_D, V_R (median over all actions).
    
    Args:
        TODO
    """
    if var_idx is not None:  # If we've supplied the VaR index we have an array of values per timestep, we need to account for this differently...
        hr_death, bins_r_death = np.histogram(np.vstack(nonsurv_data[(nonsurv_data.step==step_num)]['q_rn'].values)[:, var_idx], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hd_death, bins_d_death = np.histogram(np.vstack(nonsurv_data[(nonsurv_data.step==step_num)]['q_dn'].values)[:, var_idx], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hr_recovery, bins_r_recovery = np.histogram(np.vstack(surv_data[(surv_data.step ==step_num)]['q_rn'].values)[:, var_idx], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hd_recovery, bins_d_recovery = np.histogram(np.vstack(surv_data[(surv_data.step ==step_num)]['q_dn'].values)[:, var_idx], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hvr_death, bins_vr_death = np.histogram(np.vstack(nonsurv_data[(nonsurv_data.step==step_num)]['v_rn'].values)[:, var_idx], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hvd_death, bins_vd_death = np.histogram(np.vstack(nonsurv_data[(nonsurv_data.step==step_num)]['v_dn'].values)[:, var_idx], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hvr_recovery, bins_vr_recovery = np.histogram(np.vstack(surv_data[(surv_data.step==step_num)]['v_rn'].values)[:, var_idx], np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hvd_recovery, bins_vd_recovery = np.histogram(np.vstack(surv_data[(surv_data.step==step_num)]['v_dn'].values)[:, var_idx], np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
    else:
        hr_death, bins_r_death = np.histogram(nonsurv_data[(nonsurv_data.step==step_num)]['q_rn'].values, np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hd_death, bins_d_death = np.histogram(nonsurv_data[(nonsurv_data.step==step_num)]['q_dn'].values, np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hr_recovery, bins_r_recovery = np.histogram(surv_data[(surv_data.step ==step_num)]['q_rn'].values, np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hd_recovery, bins_d_recovery = np.histogram(surv_data[(surv_data.step ==step_num)]['q_dn'].values, np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hvr_death, bins_vr_death = np.histogram(nonsurv_data[(nonsurv_data.step==step_num)]['v_rn'].values, np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hvd_death, bins_vd_death = np.histogram(nonsurv_data[(nonsurv_data.step==step_num)]['v_dn'].values, np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)
        hvr_recovery, bins_vr_recovery = np.histogram(surv_data[(surv_data.step==step_num)]['v_rn'].values, np.linspace(0, 1, num=nb_bar+1, endpoint=True), density=False)
        hvd_recovery, bins_vd_recovery = np.histogram(surv_data[(surv_data.step==step_num)]['v_dn'].values, np.linspace(-1, 0, num=nb_bar+1, endpoint=True), density=False)

    hr_death_total = np.sum(hr_death)
    hd_death_total = np.sum(hd_death)
    hr_recovery_total = np.sum(hr_recovery)
    hd_recovery_total = np.sum(hd_recovery)
    hvr_death_total = np.sum(hvr_death)
    hvd_death_total = np.sum(hvd_death)
    hvr_recovery_total = np.sum(hvr_recovery)
    hvd_recovery_total = np.sum(hvd_recovery)

    bar_width = 1. / (nb_bar)
    w = (bar_width - (bar_width / 6)) / 2  # so that two bars fits nicely

    x = np.linspace(0, 1, num=nb_bar+1, endpoint=True)[:-1] + bar_width/2
    y = np.linspace(-1, 0, num=nb_bar+1, endpoint=True)[:-1] + bar_width/2
    rects1 = axs[0].bar(y - w/2, hd_death/hd_death_total, w, label='Non-survivors', color='blue', alpha=1) 
    rects2 = axs[0].bar(y + w/2, hd_recovery/hd_recovery_total, w, label='Survivors', color='green', alpha=1)
    rects3 = axs[1].bar(x - w/2, hr_death/hr_death_total, w, label='Non-survivors', color='blue', alpha=1)
    rects4 = axs[1].bar(x + w/2, hr_recovery/hr_recovery_total, w, label='Survivors', color='green', alpha=1)
    rects5 = axs[2].bar(y - w/2, hvd_death/hvd_death_total, w, label='Non-survivors', color='blue', alpha=1) 
    rects6 = axs[2].bar(y + w/2, hvd_recovery/hvd_recovery_total, w, label='Survivors', color='green', alpha=1)
    rects7 = axs[3].bar(x - w/2, hvr_death/hvr_death_total, w, label='Non-survivors', color='blue', alpha=1)
    rects8 = axs[3].bar(x + w/2, hvr_recovery/hvr_recovery_total, w, label='Survivors', color='green', alpha=1)


    xx = np.linspace(0, 1, num=nb_bar+1, endpoint=True)
    yy = np.linspace(-1, 0, num=nb_bar+1, endpoint=True)

    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(list(y))

    axs[0].set_xticklabels("" * len(yy - 1))

    axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_yticklabels(['0%', '50%', '100%'])

    axs[1].set_ylim(0, 1)
    axs[1].set_xticks(list(x))

    axs[1].set_xticklabels("" * len(xx - 1))

    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_yticklabels(['0%', '50%', '100%'])


    axs[2].set_ylim(0, 1)
    axs[2].set_xticks(list(y))
    
    axs[2].set_xticklabels("" * len(xx - 1))

    axs[2].set_yticks([0, 0.5, 1])
    axs[2].set_yticklabels(['0%', '50%', '100%'])

    axs[3].set_ylim(0, 1)
    axs[3].set_xticks(list(x))
    
    axs[3].set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

    axs[3].set_yticks([0, 0.5, 1])
    axs[3].set_yticklabels(['0%', '50%', '100%'])

    
    axs[0].set_title('D-Network (Q)', fontsize=7)
    axs[1].set_title('R-Network (Q)', fontsize=7)
    axs[2].set_title('D-Network (V)', fontsize=7)
    axs[3].set_title('R-Network (V)', fontsize=7)

