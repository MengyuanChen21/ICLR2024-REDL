import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics

import wandb
import pandas as pd
from PIL import Image as im

name2abbrv = {'max_prob': 'max_prob',
              'max_alpha': 'max_alpha',
              'max_modified_prob': 'max_modified_prob',
              'alpha0': 'alpha0',
              'precision': 'alpha0',
              'differential_entropy': 'diff_ent',
              'mutual_information': 'mi'}

def compute_X_Y_alpha(model, loader, device, noise_epsilon=0.0):

    X_all, Y_all, model_pred_all = [], [], []

    for batch_index, (X, Y) in enumerate(loader):
        X = (X + noise_epsilon * torch.randn_like(X)).to(device)
        Y = Y.to(device)

        model_pred = model(X, None, return_output='alpha', compute_loss=False)

        X_all.append(X.to("cpu"))
        Y_all.append(Y.to("cpu"))
        model_pred_all.append(model_pred.to("cpu"))

    X_all = torch.cat(X_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    model_pred_all = torch.cat(model_pred_all, dim=0)

    return Y_all, X_all, model_pred_all

def accuracy(Y, alpha):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).type(torch.DoubleTensor)
    accuracy = corrects.sum() / corrects.size(0)
    return accuracy.cpu().detach().numpy()


# ID detection metrics
def confidence(Y, alpha, uncertainty_type='max_prob', save_path=None, return_scores=False):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'alpha0':
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - 1.0) * (
                    torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),
                                 dim=-1)
        differential_entropy = log_term - digamma_term
        scores = - differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'mutual_information':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = - distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    if save_path is not None:
        if uncertainty_type in ['differential_entropy', 'mutual_information']:
            unc = -scores
        else:
            unc = scores

        scores_norm = (unc - min(unc)) / (max(unc) - min(unc))

        np.save(save_path, scores_norm)
        # results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(save_path)
    
    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores
    else:
        return metrics.auc(fpr, tpr)


# Our ID detection metrics
def our_confidence(Y, alpha, uncertainty_type='max_prob', save_path=None, return_scores=False, lamb1=1.0, lamb2=1.0):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_modified_prob':
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        S = evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb2 * num_classes
        p = alpha / S
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'alpha0':
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - lamb2) * (
                torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),
                                 dim=-1)
        differential_entropy = log_term - digamma_term
        scores = - differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'mutual_information':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = - distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    if save_path is not None:
        if uncertainty_type in ['differential_entropy', 'mutual_information']:
            unc = -scores
        else:
            unc = scores

        scores_norm = (unc - min(unc)) / (max(unc) - min(unc))

        np.save(save_path, scores_norm)
        # results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores
    else:
        return metrics.auc(fpr, tpr)


def brier_score(Y, alpha):
    batch_size = alpha.size(0)

    p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score


# OOD detection metrics
def anomaly_detection(alpha, ood_alpha, uncertainty_type='max_prob', save_path=None, return_scores=False):
    if uncertainty_type == 'alpha0':
        scores = alpha.sum(-1).cpu().detach().numpy()
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0].cpu().detach().numpy()

        ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        id_digamma_term = torch.sum((alpha - 1.0) * (
                    torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), dim=-1)
        id_differential_entropy = id_log_term - id_digamma_term

        ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
        ood_digamma_term = torch.sum((ood_alpha - 1.0) * (torch.digamma(ood_alpha) - torch.digamma(
            (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), dim=-1)
        ood_differential_entropy = ood_log_term - ood_digamma_term

        scores = - id_differential_entropy.cpu().detach().numpy()
        ood_scores = - ood_differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'mutual_information':
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)

        id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
        id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
        id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

        ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=1)
        ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
            ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0)
        ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)
        ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=1)
        ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty

        scores = - id_distributional_uncertainty.cpu().detach().numpy()
        ood_scores = - ood_distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        if uncertainty_type in ['differential_entropy', 'mutual_information']:
            scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))
        else:
            scores_norm = (scores - min(scores)) / (max(scores) - min(scores))

        np.save(save_path, scores_norm)
        # results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)


# OOD detection metrics for modified EDL
def our_anomaly_detection(alpha, ood_alpha, uncertainty_type='max_prob', save_path=None, return_scores=False, lamb1=1.0, lamb2=1.0):
    if uncertainty_type == 'alpha0':
        scores = alpha.sum(-1).cpu().detach().numpy()
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0].cpu().detach().numpy()

        ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_modified_prob':
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        S = evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb2 * num_classes
        p = alpha / S
        scores = p.max(-1)[0].cpu().detach().numpy()

        ood_evidence = ood_alpha - lamb2
        ood_S = ood_evidence + lamb1 * (torch.sum(ood_evidence, dim=-1, keepdim=True) - ood_evidence) + lamb2 * num_classes
        ood_p = ood_alpha / ood_S
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        id_digamma_term = torch.sum((alpha - lamb2) * (
                    torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), dim=-1)
        id_differential_entropy = id_log_term - id_digamma_term

        ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
        ood_digamma_term = torch.sum((ood_alpha - lamb2) * (torch.digamma(ood_alpha) - torch.digamma(
            (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), dim=-1)
        ood_differential_entropy = ood_log_term - ood_digamma_term

        scores = - id_differential_entropy.cpu().detach().numpy()
        ood_scores = - ood_differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'mutual_information':
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)

        id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
        id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
        id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

        ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=1)
        ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
            ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0)
        ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)
        ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=1)
        ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty

        scores = - id_distributional_uncertainty.cpu().detach().numpy()
        ood_scores = - ood_distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        if uncertainty_type in ['differential_entropy', 'mutual_information']:
            scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))
        else:
            scores_norm = (scores - min(scores)) / (max(scores) - min(scores))

        np.save(save_path, scores_norm)
        # results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)



def entropy(alpha, uncertainty_type, n_bins=10, plot=True):
    entropy = []
    if uncertainty_type == 'categorical':
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == 'dirichlet':
        entropy.append(Dirichlet(alpha).entropy().squeeze().cpu().detach().numpy())

    # if plot:
    #     plt.hist(entropy, n_bins)
    #     plt.show()
    return entropy


# additional metric based on diffEentropyUncertainty
def diff_entropy(alpha, ood_alpha, save_path=None, return_scores=False):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)

    id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
    id_digamma_term = torch.sum((alpha - 1.0) * (
                torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))), dim=-1)
    id_differential_entropy = id_log_term - id_digamma_term

    ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
    ood_digamma_term = torch.sum((ood_alpha - 1.0) * (torch.digamma(ood_alpha) - torch.digamma(
        (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha))), dim=-1)
    ood_differential_entropy = ood_log_term - ood_digamma_term

    scores = - id_differential_entropy.cpu().detach().numpy()
    ood_scores = - ood_differential_entropy.cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))

        results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)


# additional metric based on  distUncertainty
def dist_uncertainty(alpha, ood_alpha, save_path=None, return_scores=False):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)
    probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)

    id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
    id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
        alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
    id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
    id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

    ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=1)
    ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
        ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0)
    ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha)
    ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=1)
    ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty

    scores = - id_distributional_uncertainty.cpu().detach().numpy()
    ood_scores = - ood_distributional_uncertainty.cpu().detach().numpy()

    corrects = np.concatenate([np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))

        results = np.concatenate([corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1)
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)