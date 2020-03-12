import torch
import numpy as np
from time import time, sleep
from tqdm import tqdm
from torchtext import data
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import AccuracyTensorboradWriter, write_weight_statitsics, \
    ask_user_confirmation, load_checkpoint, checkpoint, vocab_to_dictionary, calc_accuracy

torch.manual_seed(1)




def drop_connect(model, rate):
    with torch.no_grad():
        dropout = torch.nn.Dropout(p=rate, inplace=True)
        for p in model.parameters():
            dropout(p)


def shift_left(tensor, padding_value, device):
    """
    tensor is 2-d, (sequence, batch)
    we are shifting the sequence and we will get (sequence+1, batch)
    """
    assert len(tensor.size()) == 2
    new_tensor = torch.full(tensor.size(), padding_value, dtype=torch.int64, device=device)
    new_tensor[:, 0:-1] = tensor[:, 1:]
    return new_tensor


def train(model, datasets, device, epochs, batch_size, pad_index, logger, content_wdecay,
          writer: SummaryWriter, checkpoint_foder, global_step=0, global_epoch=0,  verbose=True, shuffle=False):
    model.to(device)
    model = model.train()

    iterators = []
    for dataset in datasets:
        train_iter = data.BucketIterator(dataset=dataset, batch_size=batch_size,
                                         sort_key=lambda x: len(x.review), sort=False,
                                         sort_within_batch=True, repeat=False, device='cpu', shuffle=shuffle)
        iterators.append(train_iter)

    t = time()

    for e in range(epochs):

        losses = []
        accuracies = []

        for partition in tqdm(range(model.encoder.number_of_partitions)):
            model.encoder.set_partition(partition)
            active_parameters = [p for p in model.encoder.get_active_parameters()] + [p for p in model.decoder.parameters()]
            optimizer = optim.Adagrad(active_parameters)
            train_iter = iterators[partition]

            for batch in train_iter:

                batch_size = batch.id.size()[0]

                # create input
                reviews = batch.review.transpose(1, 0).to(device)

                src_tokens = torch.cat([batch.id.unsqueeze(1), batch.stars.unsqueeze(1)], dim=1).to(device)
                src_lengths = torch.full((batch_size, 1), 5).to(device)

                # run!
                model.zero_grad()
                logits, _ = model(src_tokens, src_lengths, reviews)

                logits_flat = logits.view(-1, logits.shape[-1])
                targets_flat = shift_left(reviews, pad_index, device).reshape(-1)

                embeddings = model.encoder(src_tokens, src_lengths)['encoder_out'][0]
                regloss = embeddings.norm()
                writer.add_scalar('Loss/wdecay', regloss.item(), global_step)
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_index) + regloss * content_wdecay
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.1)

                optimizer.step()

                # finished training step, now logging
                losses.append(loss.item())
                accuracies.append(calc_accuracy(logits_flat, targets_flat, ignore_index=pad_index))

                global_step += 1

        writer.add_scalar('epoch time', time() - t, global_epoch)
        t = time()
        writer.add_scalar('Loss', np.average(losses), global_epoch)
        writer.add_scalar('Acc', np.average(accuracies), global_epoch)
        logger.info('epoch {} loss {} acc {}'.format(global_epoch, np.average(losses), np.average(accuracies)))
        write_weight_statitsics(writer, model.encoder, global_epoch)
        global_epoch += 1

    checkpoint(model, os.path.join(checkpoint_foder, f'epoch{global_epoch}_checkpoint.ckpt'))
    checkpoint(model, os.path.join(checkpoint_foder, 'last_checkpoint.ckpt'))

    return global_step, global_epoch



