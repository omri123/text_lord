import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import fairseq
from model import create_model, create_partitioned_model
import os

def calc_accuracy(logits_flat, targets_flat, ignore_index):
    logits_flat_np = logits_flat.detach().cpu().numpy()
    targets_flat_np = targets_flat.detach().cpu().numpy()

    tmp1 = np.argmax(logits_flat_np, axis=1) == targets_flat_np
    tmp2 = tmp1[targets_flat_np != ignore_index]
    acc = np.sum(tmp2) / np.sum(targets_flat_np != ignore_index)

    return acc

class AccuracyTensorboradWriter:

    def __init__(self, writer: SummaryWriter, logger):
        self.writer = writer
        self.logger = logger
        self.accuracies = []
        self.accuracies10 = []

    def write_step(self, logits_flat: torch.tensor, targets_flat: torch.tensor, step: int, ignore_index:int):
        logits_flat_np = logits_flat.detach().cpu().numpy()
        targets_flat_np = targets_flat.detach().cpu().numpy()

        tmp1 = np.argmax(logits_flat_np, axis=1) == targets_flat_np
        tmp2 = tmp1[targets_flat_np != ignore_index]
        acc = np.sum(tmp2) / np.sum(targets_flat_np != ignore_index)

        self.writer.add_scalar('Acc/top-1 step', acc, step)
        self.accuracies.append(acc)

    def write_epoch(self, epoch: int):
        acc = np.average(self.accuracies)
        self.logger.info(f'finished epoch {epoch} with acc {acc} ')
        self.writer.add_scalar('Acc/top-1 epoch', acc, epoch)
        self.accuracies = []


class Accuracy10TensorboradWriter:

    def __init__(self, writer: SummaryWriter, logger):
        self.writer = writer
        self.logger = logger
        self.accuracies = []
        self.accuracies10 = []
        print("TODO!")
        assert False

    def write_step(self, logits_flat: torch.tensor, targets_flat: torch.tensor, step: int, ignore_index:int):
        logits_flat_np = logits_flat.detach().cpu().numpy()
        targets_flat_np = targets_flat.detach().cpu().numpy()

        tmp1 = np.argmax(logits_flat_np, axis=1) == targets_flat_np
        tmp2 = tmp1[targets_flat_np != ignore_index]
        acc = np.sum(tmp2) / np.sum(targets_flat_np != ignore_index)

        # acc = np.sum(np.argmax(logits_flat_np, axis=1) == targets_flat_np) / targets_flat_np.size
        # acc10 = np.sum(
        #     logits_flat_np.argsort()[:, -10:] == np.expand_dims(targets_flat_np, axis=1)) / targets_flat_np.size

        self.writer.add_scalar('Acc/top-1 step', acc, step)
        # self.writer.add_scalar('Acc/top-10 step', acc10, step)
        self.accuracies.append(acc)
        # self.accuracies10.append(acc10)

    def write_epoch(self, epoch: int):
        acc = np.average(self.accuracies)
        acc10 = np.average(self.accuracies10)
        self.logger.info(f'finished epoch {epoch} with acc {acc} and acc10 {acc10}')
        self.writer.add_scalar('Acc/top-1 epoch', acc, epoch)
        self.writer.add_scalar('Acc/top-10 epoch', acc10, epoch)
        self.accuracies = []
        self.accuracies10 = []


def write_weight_statitsics(writer: torch.utils.tensorboard.SummaryWriter, module: torch.nn.Module, epoch: int):
    # log the weights-norm for the parameters of the model
    for weight_name in module.state_dict().keys():
        w = module.state_dict()[weight_name]
        norm = w.norm().item()
        writer.add_scalar(f'Norm/{weight_name}', norm, epoch)
        avg = w.abs().mean().item()
        writer.add_scalar(f'avg/{weight_name}', avg, epoch)
        writer.add_histogram(f'hist/{weight_name}', w, epoch)


def ask_user_confirmation(msg):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(f"{msg}. OK to push to continue [Y/N]? ").lower()
    return answer == "y"


def checkpoint(model, path):
    if os.path.exists(path):
        os.remove(path)
    torch.save(model.state_dict(), path)


def load_checkpoint(path, device, *args):
    print(f'loading checkpoint {path}')
    model = create_model(*args)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model


def load_checkpoint_partitioned(path, device, *args):
    print(f'loading checkpoint {path}')
    model = create_partitioned_model(*args)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model

def vocab_to_dictionary(vocab):
    decoder_dictionary = fairseq.data.dictionary.Dictionary()
    for i in range(len(vocab)):
        decoder_dictionary.add_symbol(vocab.itos[i])
    return decoder_dictionary
