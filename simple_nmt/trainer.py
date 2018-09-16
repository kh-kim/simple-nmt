import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils


def get_loss(y, y_hat, criterion, do_backward=True):
    # |y| = (batch_size, length)
    # |y_hat| = (batch_size, length, output_size)
    batch_size = y.size(0)

    loss = criterion(y_hat.contiguous().view(-1, y_hat.size(-1)),
                     y.contiguous().view(-1)
                     )
    if do_backward:
        loss.div(batch_size).backward()

    return loss


def train_epoch(model,
                criterion,
                train_iter,
                valid_iter,
                config,
                start_epoch=1,
                others_to_save=None
                ):
    current_lr = config.lr

    lowest_valid_loss = np.inf
    no_improve_cnt = 0

    for epoch in range(start_epoch, config.n_epochs + 1):
        if config.adam:
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=current_lr)
        print("current learning rate: %f" % current_lr)
        print(optimizer)

        sample_cnt = 0
        total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
        start_time = time.time()
        train_loss = np.inf

        for batch_index, batch in enumerate(train_iter):
            # You have to reset the gradients of all model parameters before to take another step in gradient descent.
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.tgt[1])
            x = batch.src
            # Raw target variable has both BOS and EOS token. 
            # The output of sequence-to-sequence does not have BOS token. 
            # Thus, remove BOS token for reference.
            y = batch.tgt[0][:, 1:] 
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = model(x, batch.tgt[0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            # Calcuate loss and gradients with back-propagation.
            loss = get_loss(y, y_hat, criterion)
            
            # Simple math to show stats.
            total_loss += float(loss)
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            # Print current training status in every this number of mini-batch is done.
            if (batch_index + 1) % config.print_every == 0:
                avg_loss = total_loss / total_word_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                elapsed_time = time.time() - start_time

                # You can check the current status using parameter norm and gradient norm.
                # Also, you can check the speed of the training.
                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\tloss: %.4f\tPPL: %.2f\t%5d words/s %3d secs" % (epoch,
                                                                                                                               batch_index + 1,
                                                                                                                               int(len(train_iter.dataset.examples) // config.batch_size),
                                                                                                                               avg_parameter_norm,
                                                                                                                               avg_grad_norm,
                                                                                                                               avg_loss,
                                                                                                                               np.exp(avg_loss),
                                                                                                                               total_word_count // elapsed_time,
                                                                                                                               elapsed_time
                                                                                                                               ))

                total_loss, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0
                start_time = time.time()

                train_loss = avg_loss

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(),
                                        config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch.tgt[0].size(0)
            if sample_cnt >= len(train_iter.dataset.examples):
                break

        sample_cnt = 0
        total_loss, total_word_count = 0, 0

        with torch.no_grad():  # In validation, we don't need to get gradients.
            model.eval()  # Turn-on the evaluation mode.

            for batch_index, batch in enumerate(valid_iter):
                current_batch_word_cnt = torch.sum(batch.tgt[1])
                x = batch.src
                y = batch.tgt[0][:, 1:]
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)

                # Take feed-forward
                y_hat = model(x, batch.tgt[0][:, :-1])
                # |y_hat| = (batch_size, length, output_size)

                loss = get_loss(y, y_hat, criterion, do_backward=False)

                total_loss += float(loss)
                total_word_count += int(current_batch_word_cnt)

                sample_cnt += batch.tgt[0].size(0)
                if sample_cnt >= len(valid_iter.dataset.examples):
                    break

            # Print result of validation.
            avg_loss = total_loss / total_word_count
            print("valid loss: %.4f\tPPL: %.2f" % (avg_loss, np.exp(avg_loss)))

            if lowest_valid_loss > avg_loss:
                lowest_valid_loss = avg_loss
                no_improve_cnt = 0

                # Altough there is an improvement in last epoch, we need to decay the learning-rate if it meets the requirements.
                if epoch >= config.lr_decay_start_at:
                    current_lr = max(config.min_lr,
                                     current_lr * config.lr_decay_rate
                                     )
            else:
                # Decrease learing rate if there is no improvement.
                current_lr = max(config.min_lr,
                                 current_lr * config.lr_decay_rate
                                 )
                no_improve_cnt += 1

            # Again, turn-on the training mode.
            model.train()

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % epoch,
                                    "%.2f-%.2f" % (train_loss, np.exp(train_loss)),
                                    "%.2f-%.2f" % (avg_loss, np.exp(avg_loss))
                                    ] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        to_save = {"model": model.state_dict(),
                   "config": config,
                   "epoch": epoch + 1,
                   "current_lr": current_lr
                   }
        if others_to_save is not None:  # Add others if it is necessary.
            for k, v in others_to_save.items():
                to_save[k] = v
        torch.save(to_save, '.'.join(model_fn))

        # Take early stopping if it meets the requirement.
        if config.early_stop > 0 and no_improve_cnt > config.early_stop:
            break
