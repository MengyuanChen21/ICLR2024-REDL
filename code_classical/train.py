import torch
import numpy as np
import wandb

def compute_loss_accuracy(model, loader, epoch, device=torch.device("cpu"), is_fisher=False):
    model.eval()
    with torch.no_grad():
        total_loss_ = 0.0

        if is_fisher:
            loss_mse_ = 0.0
            loss_var_ = 0.0
            loss_kl_ = 0.0
            loss_fisher_ = 0.0

        for batch_index, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)

            if model.loss == 'DUQ':
                Y_pred = model(X, Y, return_output='hard', compute_loss=False, epoch=epoch)
            else:
                Y_pred = model(X, Y, return_output='hard', compute_loss=True, epoch=epoch)

            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)

            total_loss_ += model.grad_loss.item()

            if is_fisher:
                loss_mse_ += model.loss_mse_.item()
                loss_var_ += model.loss_var_.item()
                loss_kl_ += model.loss_kl_.item()
                loss_fisher_ += model.loss_fisher_.item()

        total_loss_ = total_loss_ / Y_pred_all.size(0)
        if is_fisher:
            loss_mse_ = loss_mse_ / Y_pred_all.size(0)
            loss_var_ = loss_var_ / Y_pred_all.size(0)
            loss_kl_ = loss_kl_ / Y_pred_all.size(0)
            loss_fisher_ = loss_fisher_ / Y_pred_all.size(0)

        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()

    model.train()
    if is_fisher:
        return accuracy, total_loss_, loss_mse_, loss_var_, loss_kl_, loss_fisher_
    else:
        return accuracy, total_loss_


def train(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=5, model_path='saved_model',
          full_config_dict={}, use_wandb=False, device=torch.device("cpu"), is_fisher=False, output_dim=10):
    model.to(device)
    model.train()
    val_losses, val_accuracies = [], []
    best_val_loss = float("Inf")
    val_loss = float("Inf")
    best_val_acc = 0.0
    epoch = 0

    for epoch in range(max_epochs):
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            model.train()
            model(X_train, Y_train, compute_loss=True, epoch=epoch)
            model.step()
            # model.module.step()

        if full_config_dict['dataset_name'] == 'MNIST' or full_config_dict['model_type'] == 'DUQ':
            model.scheduler.step()

        if epoch % frequency == 0:
            # Stats on datasets sets
            if use_wandb:
                if is_fisher:
                    train_accuracy, total_loss_, loss_mse_, loss_var_, loss_kl_, loss_fisher_ = compute_loss_accuracy(
                        model, train_loader, epoch, device=device, is_fisher=True)
                    wandb.log({'Train/total_loss_': round(total_loss_, 3), 'Train/loss_mse_': round(loss_mse_, 3),
                               'Train/loss_var_': round(loss_var_, 3), 'Train/loss_kl_': round(loss_kl_, 3),
                               'Train/loss_fisher_': round(loss_fisher_, 3), 'Train/Acc': round(train_accuracy * 100, 3),
                               'Train/epoch': epoch + 1})
                else:
                    train_accuracy, total_loss_ = compute_loss_accuracy(model, train_loader, epoch, device=device)
                    wandb.log({'Train/total_loss_': round(total_loss_, 3), 'Train/Acc': round(train_accuracy * 100, 3),
                               'Train/epoch': epoch + 1})

            val_accuracy, val_loss = compute_loss_accuracy(model, val_loader, epoch, device=device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if use_wandb:
                wandb.log({'Val/total_loss_': round(val_loss * 100, 3), 'Val/Acc': round(val_accuracy * 100, 3),
                           'Val/epoch': epoch + 1})

            print("\033[34m Epoch {} -> Val loss {} | Val Acc. {}% | Best Val Acc. {}%\033[0m".format(
                epoch,
                round(val_losses[-1] * 100, 3),
                round(val_accuracies[-1] * 100, 3),
                round(best_val_acc * 100, 3)))

            if best_val_acc < val_accuracies[-1]:
                best_val_acc = val_accuracies[-1]
                torch.save(
                    {'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(),
                     'loss': best_val_loss}, f"{model_path}_best")
                print(f'Best model saved, Epoch: {epoch}')

            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            # if int(epoch / frequency) > patience and val_accuracies[-patience] >= max(val_accuracies[-patience:]):
            #     print('Early Stopping.')
            #     break

    torch.save(
        {'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(),
         'loss': val_loss}, f"{model_path}_last")
    print(f'Last model saved, Epoch: {epoch}')

    return
