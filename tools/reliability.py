"""
Tools for plotting calibration figures (accuracy vs confidence).
    1. accuracy histogram
    2. confidence histogram
    3. reliability(accuracy vs confidence) diagram
Thanks to reference code : https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


def confidence_histogram(bin_data,
                         draw_mean=True,
                         xlabel="Confidence",        
                         ylabel="Count",               
                         save_name=None):
    """Draws a accuracy histogram."""    
    confidences = bin_data["confidences"]   
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0
    sns.set()    
    fig = plt.figure()
    ax = fig.subplots()
    ax.bar(positions, counts, width=bin_size*1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if draw_mean:
        avg_accuracy = bin_data["avg_accuracy"]
        avg_confidence = bin_data["avg_confidence"]
        acc_plt = ax.axvline(avg_accuracy, ls="solid", lw=3 ,label="Accuracy", c="black")
        conf_plt = ax.axvline(avg_confidence, ls="dotted", lw=3 ,label="Avg. confidence", c="black")
        ax.legend(handles=[acc_plt, conf_plt])
    if save_name is not None:
        plt.savefig(save_name)
    plt.clf()


def reliability_diagram(bin_data,
                        bin_num=20,
                        xlabel="Confidence", 
                        ylabel="Accuracy",
                        save_name=None):
    """Draws a reliability diagram."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]
    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    sns.set()
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(positions, accuracies, marker="o", c="red", label="Accuracy")
    ax.plot(np.linspace(0,1), np.linspace(0,1), ":k")
    ax.bar(positions, counts/np.sum(counts), width=bin_size*1, label="% of Samples")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Confidence")    
    ax.legend()

    fig.tight_layout()

    #p = sns.jointplot(confidences, accuracies, marginal_kws=dict(bins=bin_num, rug=True),xlim=(0,1), ylim=(0,1))        
    #p.set_axis_labels(xlabel, ylabel, fontsize=12)
    if save_name is not None:
        plt.savefig(save_name)
    plt.close()

def compute_calibration(pred_labels, true_labels, confidences, num_bins):
    """
    Compute calibration measures
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    pred_labels = np.array(pred_labels, np.float)
    true_labels = np.array(true_labels, np.float)
    confidences = np.array(confidences, np.float)

    bins = np.linspace(0.0, 1.0, num_bins+1)
    indices = np.digitize(confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.float)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    oe = np.sum(np.maximum(bin_confidences - bin_accuracies, np.zeros_like(gaps)) * bin_counts) / np.sum(bin_counts)
    return { "accuracies": bin_accuracies,
             "confidences": bin_confidences,
             "counts": bin_counts,
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce,
             "overconfidence_error": oe,
             }

if __name__ == "__main__":
    print('Testing code..')
    DATA_NUM = 2000
    BIN_NUM = 100
    PRED = np.random.randint(10, size=DATA_NUM)
    LABEL = np.random.randint(10, size=DATA_NUM)
    PROB = np.random.rand(DATA_NUM)
    INFOS = compute_calibration(PRED, LABEL, PROB, BIN_NUM)
    print(INFOS['accuracies'], INFOS['confidences'], INFOS['counts'])
    print(INFOS['max_calibration_error'])
    print(INFOS['overconfidence_error'])
    print(INFOS['bins'])
    reliability_diagram(INFOS)
    confidence_histogram(INFOS)