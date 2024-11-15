import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # load data
    best_file = '../results/long_term_forecast_ElcPrice_96_96_Informer_ElcPrice_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_16_0'
    pred_path = os.path.join(best_file, 'pred.npy')
    true_path = os.path.join(best_file, 'true.npy')
    pred = np.load(pred_path)
    true = np.load(true_path)
    pre_ = []
    tru_ = []
    for i in range(0, 21):
        pre_.append(pred[i*96])
        tru_.append(true[i*96])
    preds = np.concatenate(pre_, axis=0)
    trues = np.concatenate(tru_, axis=0)
    # plot 21 days
    plt.figure()
    plt.plot(preds.reshape(-1, 1), label='Prediction')
    plt.plot(trues.reshape(-1, 1), label='GroundTruth')
    plt.legend()
    # save figure
    plt.savefig(os.path.join(best_file, '21days.png'), bbox_inches='tight', dpi=300)