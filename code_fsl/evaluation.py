import torch
from metrics import accuracy, confidence, anomaly_detection, diff_entropy, dist_uncertainty

name2abbrv = {'lamb_max_prob': 'lamb_max_prob',
              'max_prob': 'max_prob',
              'max_alpha': 'max_alpha',
              'alpha0': 'alpha0',
              'differential_entropy': 'diff_ent',
              'distribution_uncertainty': 'mi'}


def compute_output(model, inputs, act_type='exp'):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

        if act_type == 'exp':
            alpha = torch.exp(outputs) + 1.0
        elif act_type == 'softplus':
            evi = torch.nn.Softplus()
            alpha = evi(outputs) + 1.0
        elif act_type == 'relu':
            alpha = torch.relu(outputs) + 1.0
        else:
            alpha = outputs

        return alpha


def test_misclassication(model, act_type, id_x, id_y):
    with torch.no_grad():
        metrics = {}

        id_alpha = compute_output(model, id_x, act_type=act_type)

        # Save metrics
        metrics['id_accuracy'] = accuracy(Y=id_y, alpha=id_alpha).tolist()

        for name in ['max_prob', 'lamb_max_prob', 'max_alpha', 'alpha0', 'differential_entropy', 'distribution_uncertainty']:
            abb_name = name2abbrv[name]
            metrics[f'id_{abb_name}_apr'] = confidence(Y=id_y, alpha=id_alpha, score_type='APR', uncertainty_type=name)
            metrics[f'id_{abb_name}_auroc'] = confidence(Y=id_y, alpha=id_alpha, score_type='AUROC',
                                                         uncertainty_type=name)

    return metrics


def test_ood_uncertainty(model, act_type, id_x, ood_x, ood_y):
    with torch.no_grad():
        metrics = {}

        _, n_samps_id, _ = id_x.shape
        _, n_samps_ood, _ = ood_x.shape

        n_samps = min(n_samps_id, n_samps_ood)

        id_alpha = compute_output(model, id_x[:, :n_samps, :], act_type=act_type)
        ood_alpha = compute_output(model, ood_x[:, :n_samps, :], act_type=act_type)

        # metrics['ood_accuracy'] = accuracy(Y=ood_y, alpha=ood_alpha).tolist()

        # for name in ['max_prob', 'precision']:
        for name in ['max_prob', 'lamb_max_prob', 'max_alpha', 'alpha0']:

            abb_name = name2abbrv[name]

            metrics[f'ood_{abb_name}_apr'] = anomaly_detection(alpha=id_alpha, ood_alpha=ood_alpha, score_type='APR',
                                                               uncertainty_type=name)
            metrics[f'ood_{abb_name}_auroc'] = anomaly_detection(alpha=id_alpha, ood_alpha=ood_alpha,
                                                                 score_type='AUROC', uncertainty_type=name)

        metrics['ood_diff_entropy_apr'] = diff_entropy(alpha=id_alpha, ood_alpha=ood_alpha, score_type='APR')
        metrics['ood_mi_apr'] = dist_uncertainty(alpha=id_alpha, ood_alpha=ood_alpha, score_type='APR')
        metrics['ood_diff_entropy_auroc'] = diff_entropy(alpha=id_alpha, ood_alpha=ood_alpha, score_type='AUROC')
        metrics['ood_mi_auroc'] = dist_uncertainty(alpha=id_alpha, ood_alpha=ood_alpha, score_type='AUROC')

    return metrics
