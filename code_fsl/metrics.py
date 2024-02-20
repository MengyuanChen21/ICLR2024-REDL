import torch
import numpy as np
from sklearn import metrics


def accuracy(Y, alpha):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).type(torch.DoubleTensor)
    return corrects.mean(-1).detach().cpu().numpy().ravel()


def brier_score(Y, alpha):
    # alpha.shape -> batch_dim, n_samples, n_ways
    Y_1hot_ = torch.zeros_like(alpha).scatter_(-1, Y.unsqueeze(-1), 1)
    alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
    probs = alpha / alpha0

    brier_score = (probs - Y_1hot_).norm(dim=-1).mean(-1).cpu().detach().tolist()
    return brier_score


def confidence(Y, alpha, score_type='AUROC', uncertainty_type='aleatoric', lamb1=1.0, lamb2=1.0):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == 'alpha0':
        scores = alpha.sum(-1)
    elif uncertainty_type == 'lamb_max_prob':
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        prob = alpha / (evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence)
                                 + lamb2 * num_classes)
        scores = prob.max(-1)[0]
    elif uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0]
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0]
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(alpha0)
        digamma_term = torch.sum((alpha - lamb2) * (torch.digamma(alpha) - torch.digamma(alpha0)), dim=-1, keepdim=True)
        differential_entropy = (log_term - digamma_term).sum(-1)
        scores = - differential_entropy
    elif uncertainty_type == 'distribution_uncertainty':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / alpha0

        total_uncertainty = - torch.sum(probs * torch.log(probs + 0.00001), dim=-1, keepdim=True)

        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 + 1.0)
        dirichlet_mean = alpha / alpha0
        exp_data_uncertainty = - torch.sum(dirichlet_mean * digamma_term, dim=-1, keepdim=True)

        distributional_uncertainty = (total_uncertainty - exp_data_uncertainty).sum(-1)
        scores = - distributional_uncertainty
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    scores = scores.cpu().detach().numpy()

    assert corrects.shape == scores.shape and len(corrects.shape) == 2

    value_list = []
    for task_id in range(len(alpha)):
        if score_type == 'AUROC':
            fpr, tpr, thresholds = metrics.roc_curve(corrects[task_id], scores[task_id])
            value_list.append(metrics.auc(fpr, tpr))
        elif score_type == 'APR':
            value_list.append(metrics.average_precision_score(corrects[task_id], scores[task_id]))
        else:
            raise ValueError(f"Invalid score type: {score_type}!")

    return value_list


# OOD detection metrics
def anomaly_detection(alpha, ood_alpha, score_type='AUROC', uncertainty_type='aleatoric', lamb1=1.0, lamb2=1.0):
    if uncertainty_type == 'alpha0':
        scores = alpha.sum(-1)
    elif uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0]
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0]
    elif uncertainty_type == 'lamb_max_prob':
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        prob = alpha / (evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence)
                                     + lamb2 * num_classes)
        scores = prob.max(-1)[0]
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    scores = scores.cpu().detach().numpy()

    if uncertainty_type == 'alpha0':
        ood_scores = ood_alpha.sum(-1)
    elif uncertainty_type == 'max_alpha':
        ood_scores = ood_alpha.max(-1)[0]
    elif uncertainty_type == 'max_prob':
        p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
        ood_scores = p.max(-1)[0]
    elif uncertainty_type == 'lamb_max_prob':
        num_classes = ood_alpha.shape[-1]
        ood_evidence = ood_alpha - lamb2
        ood_prob = ood_alpha / (ood_evidence + lamb1 * (torch.sum(ood_evidence, dim=-1, keepdim=True) - ood_evidence)
                                         + lamb2 * num_classes)
        ood_scores = ood_prob.max(-1)[0]
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    ood_scores = torch.where(torch.isfinite(ood_scores), ood_scores, torch.zeros_like(ood_scores))
    ood_scores = ood_scores.cpu().detach().numpy()

    assert alpha.shape == ood_alpha.shape
    batch_dim, n_samps, _ = alpha.shape
    corrects = np.concatenate([np.ones((batch_dim, n_samps)), np.zeros((batch_dim, n_samps))], axis=-1)
    scores = np.concatenate([scores, ood_scores], axis=-1)

    assert corrects.shape == scores.shape and len(corrects.shape) == 2

    value_list = []
    for task_id in range(len(alpha)):
        if score_type == 'AUROC':
            fpr, tpr, thresholds = metrics.roc_curve(corrects[task_id], scores[task_id])
            value_list.append(metrics.auc(fpr, tpr))
        elif score_type == 'APR':
            value_list.append(metrics.average_precision_score(corrects[task_id], scores[task_id]))
        else:
            raise ValueError(f"Invalid score type: {score_type}!")

    return value_list


# additional metric based on diffEentropyUncertainty
def diff_entropy(alpha, ood_alpha, score_type='AUROC', lamb2=1.0):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
    ood_alpha0 = torch.sum(ood_alpha, dim=-1, keepdim=True)

    id_log_term = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(alpha0)
    id_digamma_term = torch.sum((alpha - lamb2) * (torch.digamma(alpha) - torch.digamma(alpha0)), dim=-1, keepdim=True)
    id_differential_entropy = (id_log_term - id_digamma_term).sum(-1)

    ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1, keepdim=True) - torch.lgamma(ood_alpha0)
    ood_digamma_term = torch.sum((ood_alpha - lamb2) * (torch.digamma(ood_alpha) - torch.digamma(ood_alpha0)), dim=-1, keepdim=True)
    ood_differential_entropy = (ood_log_term - ood_digamma_term).sum(-1)

    scores = - id_differential_entropy
    ood_scores = - ood_differential_entropy

    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores)).cpu().detach().numpy()
    ood_scores = torch.where(torch.isfinite(ood_scores), ood_scores, torch.zeros_like(ood_scores)).cpu().detach().numpy()

    assert alpha.shape == ood_alpha.shape
    batch_dim, n_samps, _ = alpha.shape
    corrects = np.concatenate([np.ones((batch_dim, n_samps)), np.zeros((batch_dim, n_samps))], axis=-1)
    scores = np.concatenate([scores, ood_scores], axis=-1)

    assert corrects.shape == scores.shape and len(corrects.shape) == 2
    value_list = []
    for task_id in range(len(alpha)):
        if score_type == 'AUROC':
            fpr, tpr, thresholds = metrics.roc_curve(corrects[task_id], scores[task_id])
            value_list.append(metrics.auc(fpr, tpr))
        elif score_type == 'APR':
            value_list.append(metrics.average_precision_score(corrects[task_id], scores[task_id]))
        else:
            raise ValueError(f"Invalid score type: {score_type}!")

    return value_list


# additional metric based on  distUncertainty
def dist_uncertainty(alpha, ood_alpha, score_type='AUROC'):
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
    ood_alpha0 = torch.sum(ood_alpha, dim=-1, keepdim=True)
    probs = alpha / alpha0
    ood_probs = ood_alpha / ood_alpha0

    id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1, keepdim=True)
    id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(alpha0 +1.0)
    id_dirichlet_mean = alpha / alpha0
    id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=-1, keepdim=True)
    id_distributional_uncertainty = (id_total_uncertainty - id_exp_data_uncertainty).sum(-1)

    ood_total_uncertainty = -1 * torch.sum(ood_probs * torch.log(ood_probs + 0.00001), dim=-1, keepdim=True)
    ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(ood_alpha0 + 1.0)
    ood_dirichlet_mean = ood_alpha / ood_alpha0
    ood_exp_data_uncertainty = -1 * torch.sum(ood_dirichlet_mean * ood_digamma_term, dim=-1, keepdim=True)
    ood_distributional_uncertainty = (ood_total_uncertainty - ood_exp_data_uncertainty).sum(-1)

    scores = - id_distributional_uncertainty
    ood_scores = - ood_distributional_uncertainty

    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores)).cpu().detach().numpy()
    ood_scores = torch.where(torch.isfinite(ood_scores), ood_scores,
                             torch.zeros_like(ood_scores)).cpu().detach().numpy()

    assert alpha.shape == ood_alpha.shape
    batch_dim, n_samps, _ = alpha.shape

    corrects = np.concatenate([np.ones((batch_dim, n_samps)), np.zeros((batch_dim, n_samps))], axis=-1)
    scores = np.concatenate([scores, ood_scores], axis=-1)

    assert corrects.shape == scores.shape and len(corrects.shape) == 2
    value_list = []
    for task_id in range(len(alpha)):
        if score_type == 'AUROC':
            fpr, tpr, thresholds = metrics.roc_curve(corrects[task_id], scores[task_id])
            value_list.append(metrics.auc(fpr, tpr))
        elif score_type == 'APR':
            value_list.append(metrics.average_precision_score(corrects[task_id], scores[task_id]))
        else:
            raise ValueError(f"Invalid score type: {score_type}!")

    return value_list
