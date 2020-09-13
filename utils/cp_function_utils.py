import sklearn.metrics
import numpy as np
import torch
import torch.nn.functional as F
import os, sys, os.path as osp

def roc_score_onehot(lpred, ltrue, all_wlabel=None):
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(ltrue, lpred, sample_weight=all_wlabel)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    return auc

def pr_score_onehot(lpred, ltrue):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ltrue, lpred)
    auc = sklearn.metrics.auc(recall, precision)
    return auc

def make_vector_label(target, ll, w=None):
    l_label = np.zeros(ll)
    if w is None:
        l_label[target] = 1
        w_label=None
    else:
        w_label = np.ones(ll)
        w_label[target-w:target] = np.arange(w) * (1/(w+1))
        w_label[target+1:target+w+1] = w_label[target-w:target][::-1]
        l_label[target-w: target+w+1] = 1
    return l_label, w_label

def roc_score(res, cpd, metric=roc_score_onehot, burn_in=25, w=None):
    n = len(res)
    burn_in # require burn in time steps before making a prediction
    all_res = []
    all_label = []
    all_wlabel = []
    for i in range(n):
        cur_res = res[i]
        l = cpd[i]
        ll = len(cur_res)
        cur_res = cur_res[burn_in:ll-burn_in]
        l_label, w_label = make_vector_label(l, ll, w)
        l_label = l_label[burn_in:ll-burn_in]
        if w_label is not None:
            w_label = w_label[burn_in:ll-burn_in]
            all_wlabel.append(w_label)
        all_res.append(cur_res)
        all_label.append(l_label)

    all_res = np.concatenate(all_res, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    if len(all_wlabel) > 0:
        all_wlabel = np.concatenate(all_wlabel, axis=0)
        return metric(all_res, all_label, all_wlabel)
    else:
        return metric(all_res, all_label)

# average distance and triangle utility for evaluation
def top_one_anomaly(res, burn_in=25):
    return burn_in + np.argmax(res[..., burn_in:-burn_in], axis=-1)

def avg_distance(res, cpd, burn_in=25):
    pred = top_one_anomaly(res, burn_in)
    return np.mean(abs(pred - cpd))

def triangle_utility(x, y, w):
    val = 1 - abs(x - y) / w
    return np.where(val>0, val, 0)

def avg_triangle_utility(res, cpd, w=5, burn_in=25):
    pred = top_one_anomaly(res, burn_in)
    return np.mean(triangle_utility(pred, cpd, w))


def cal_cp(pred):
    pred = np.argmax(pred, -1)
    score = np.diff(pred, axis=0)
    score = np.sum(abs(score), axis=-1)
    score = [0] + list(score)
    return np.array(score)

def cal_cp_continuous(pred):
    pred = pred[..., 1]
    score = np.diff(pred, axis=0)
    score = np.sum(abs(score), axis=-1)
    score = [0] + list(score)
    return np.array(score)


def cal_cp_mean(pred):
    pred_mean = []
    ll = len(pred)
    w = 5
    for i in range(w - 1, ll):
        pred_mean.append(np.mean(pred[i - w + 1:i + 1], 0))
    pred_mean = np.array(pred_mean)
    pred_mean = np.argmax(pred_mean, -1)
    score = np.diff(pred_mean, axis=0)
    score = np.sum(abs(score), axis=-1)
    score = [0] * w + list(score)
    return np.array(score)


def cal_cp_from_output(output, fun=cal_cp_continuous):
    n = len(output)
    scores = []
    for i in range(n):
        scores.append(fun(output[i]))
    return np.array(scores)

def cpd_metrics(output, cpd, anomaly_input=False, tri_w=15, roc_w=1, burn_in=25):
    if anomaly_input:
        res = output
    else:
        res = cal_cp_from_output(output, cal_cp_continuous)
    avg_dist = avg_distance(res, cpd, burn_in=burn_in)
    avg_tri = avg_triangle_utility(res, cpd, w=tri_w, burn_in=burn_in)
    avg_roc = roc_score(res, cpd, burn_in=burn_in, w=roc_w)
    return avg_roc, avg_dist, avg_tri


# anomaly analysis
def plot_anomaly(lpred, ltrue, w=10, save_name=None):
    ll = len(lpred)
    #lpred = softmax(lpred)
    
    plt.plot(np.arange(ll - 2*w) + w, lpred[w:-w], color='b', label='pred')
    plt.axvline(ltrue, color='r', linestyle='dashed', label='true')
    plt.legend()
    #plt.xticks(np.arange(25, 175, 10))
    if save_name is not None:
        plt.savefig("fig/{}".format(save_name))
    plt.show()
    
def value_from_folder(model_path, mode='valid', data_type=None):
    print("files in exp: ", os.listdir(model_path))
    if data_type is None:
        npy_file = "{}_dict.npy".format(mode)
    else:
        npy_file = "{}_{}_dict.npy".format(data_type, mode)
    dic = np.load(osp.join(model_path, npy_file)).item()

    cpds = dic["cpds"]
    probs = dic["probs"]
    relations = dic["relations"]
    recons = dic["recons"]
    origs = dic["origs"]
    
    correlation_score = cal_cp_from_output(probs, cal_cp_continuous)
    independent_score = mse_anomaly(recons, origs, step=5)
    
    return (correlation_score, independent_score), (probs, cpds, relations, (origs, recons))

# anomlay mse prediction
def batched_mean_squared_error(pred, target):
    raw = (pred - target)**2
    raw = raw.reshape((raw.shape[0], -1))
    return np.mean(raw, -1)

def mse_anomaly(recons, origs, step=1):
    # recons [cases, time_steps, pred_steps, num_obj, dim]
    # origs [cases, time_steps, num_obj, dim]
    cases, time_steps, num_obj, dim = origs.shape
    cases, recon_time_steps, pred_step, num_obj, dim = recons.shape
    origs = origs.reshape((cases, time_steps, -1))
    recons = recons.reshape((cases, recon_time_steps, pred_step, -1))
    anomaly_mse = np.zeros((cases, time_steps))
    for cur in range(1, time_steps-pred_step):
        anomaly_mse[:, cur] = np.squeeze(batched_mean_squared_error((recons[:, cur-1, :step]), origs[:, cur:cur+step]))
    return anomaly_mse

# uniformed anomlay score
def norm_score_min_max(score):
    def _transform(s):
        mi, ma = s.min(), s.max()
        return (s - mi) / (ma - mi)
    return np.array([_transform(s) for s in score])
def norm_score_mean_std(score):
    def _transform(s):
        u, st = np.mean(s), np.std(s)
        return (s - u) / st
    return np.array([_transform(s) for s in score])

def anomaly_combined_score(a1, a2, fun=norm_score_mean_std):
    if np.sum(abs(a1)) > 0:
        a1 = fun(a1)
    else:
        a1 = 0
    if np.sum(abs(a2)) > 0:
        a2 = fun(a2)
    else:
        a2 = 0
    combine = a1 + a2
    return combine

def unsupervised_prediction(a1, a2, fun=norm_score_mean_std):
    combine = anomaly_combined_score(a1, a2, fun)
    return top_one_anomaly(combine)

def change_type_classification(a1, a2, supervision=None, fun=None):
    if supervision is None:
        pred = unsupervised_prediction(a1, a2)
    else:
        pred = supervision
    n = pred.shape[0]
    if fun is not None:
        a1, a2 = fun(a1), fun(a2)
    pred_type = (a1[np.arange(n), pred] - a2[np.arange(n), pred]) > 0
    type1 = np.sum(pred_type)
    return type1, n-type1

