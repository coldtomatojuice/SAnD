import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from utils.trainer import NeuralNetworkClassifier

import logging
import logging.handlers


class run():
    def __init__(self, log_file:str):
        self.logger = self.get_logger(log_file=log_file)
        self.train_loader, self.val_loader, self.test_loader = self.gen_pseudo_dataset()

    def get_logger(self, log_file:str):
        logger = logging.getLogger(__name__)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')

        fileHandler = logging.FileHandler(filename=log_file)
        fileHandler.setFormatter(formatter)

        logger.addHandler(fileHandler)

        logger.setLevel(level=logging.DEBUG)
        
        return logger

    def gen_pseudo_dataset(self):
        x_train = torch.randn(2048, 256, 23)    # [N, seq_len, features]
        x_val = torch.randn(128, 256, 23)       # [N, seq_len, features]
        x_test =  torch.randn(512, 256, 23)     # [N, seq_len, features]

        y_train = torch.randint(0, 9, (2048, ))
        y_val = torch.randint(0, 9, (128, ))
        y_test = torch.randint(0, 9, (512, ))

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=128)
        val_loader = DataLoader(val_ds, batch_size=128)
        test_loader = DataLoader(test_ds, batch_size=128)

        return train_loader, val_loader, test_loader

    def train(self):
        in_feature = 23
        seq_len = 256
        n_heads = 32
        factor = 32
        num_class = 10
        num_layers = 6

        # optimizer_config = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}
        optimizer_config = {"lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4}

        clf = NeuralNetworkClassifier(
            SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
            nn.CrossEntropyLoss(),
            optim.Adam, 
            optimizer_config=optimizer_config,
            logger=self.logger
        )

        clf.fit(
            {"train": self.train_loader, "val": self.val_loader},
            epochs=200
        )

        clf.evaluate(self.test_loader)
        clf.confusion_matrix(self.test_loader.dataset)
        clf.save_to_file("save_params/")
