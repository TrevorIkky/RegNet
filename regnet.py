import torch
import typing
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F

import argparse

import pytorch_lightning as pl
from conv_rnns import ConvGRUCell, ConvLSTMCell


from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from cifar10_datamodule import Cifar10DataModule

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

momentum = 0.9
max_epochs = 30
batch_size = 64

class rnn_regulated_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, rnn_cell=None, identity_block=None, stride=1):
        super(rnn_regulated_block, self).__init__()
        #print(f'In channels {in_channels} | Intermediate channels: {intermediate_channels} ')
        self.identity_block = identity_block
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()

        self.rnn_cell = rnn_cell
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.LazyConv2d(intermediate_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(intermediate_channels * 4)
    def forward(self, x:torch.Tensor, state:typing.Tuple) -> typing.Tuple:
        c, h = state
        y = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.rnn_cell is not None:
            if isinstance(self.rnn_cell, ConvLSTMCell):
                c, h = self.rnn_cell(x, state)
            else:
                c = None; h = self.rnn_cell(x, state[1])

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print(f'Block running. x.shape : {x.shape}, h shape: {h.shape}')
        x = torch.cat([x, h], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        if self.identity_block is not None:
            y = self.identity_block(y)
        x += y
        return c, h, self.relu(x)

class RegNet(pl.LightningModule):
    def __init__(self, regulated_block:nn.Module, in_dim:int, intermediate_channels:int, classes:int=3, cell_type:str='gru', layers:typing.List=[3, 4, 6, 3], config=None):
        super(RegNet, self).__init__()
        self.layers = layers
        self.classes = classes
        self.intermediate_channels = intermediate_channels
        #self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 3) , padding=1, stride=2)
        self.cell = ConvGRUCell if cell_type == 'gru' else ConvLSTMCell
        regulated_blocks = []

        for layer in range(len(layers)):
            stride = 1 if layer < 1 else 2
            channels = self.intermediate_channels if layer < 1 else self.intermediate_channels // 2
            h_channels = intermediate_channels
            identity_block = nn.Sequential(
                nn.Conv2d(self.intermediate_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )

            regulated_blocks.append(regulated_block(
                self.intermediate_channels, channels,
                self.cell(channels,h_channels , kernel_size=3),
                identity_block, stride
            ))

            for block in range(layers[layer] - 1):
                self.intermediate_channels = channels * 4 if block < 1 else self.intermediate_channels
                regulated_blocks.append(regulated_block(
                        self.intermediate_channels, channels
                    )
                )

        self.state_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.regulated_blocks = nn.ModuleList(regulated_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(classes)

        self.val_accuracy = tm.Accuracy()
        self.test_accuracy = tm.Accuracy()
        self.train_accuracy = tm.Accuracy()

        self.config = config

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.max_pool(x)
        c, h = torch.zeros(x.shape), torch.zeros(x.shape)
        for _, block in enumerate(self.regulated_blocks):
            #print(f'Block: {i}, x.shape: {x.shape}, h.shape {h.shape}')
            c, h, x = block(x, (c, h))
            if h.shape[-1] != x.shape[-1]:
                h = self.state_max_pool(h)
                if c is not None:
                    c = self.state_max_pool(c)

        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.output(x)


    def configure_optimizers(self):
        learning_rate = 0.1
        weight_decay = 5e-4

        if self.config is not None:
            learning_rate = self.config['lr']
            weight_decay = self.config['weight_decay']

        optimizer= SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return { "optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor":  "val_accuracy"}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.train_accuracy(outputs, labels)
        return { "loss" : loss, "accuracy" : accuracy }

    def training_epoch_end(self, outputs):
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.val_accuracy(outputs, labels)
        return { "val_loss" : loss }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)


    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = self.test_accuracy(outputs, labels)
        return { "test_loss" : loss }

    def test_epoch_end(self, outputs):
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)


def train_regnet(config, num_epochs=10, num_gpus=1):

    block1 = config['block1']
    block2 = config['block2']
    block3 = config['block3']
    block4 = config['block4']
    layers = [block1, block2, block3, block4]

    batch_size = config['batch_size']
    intermediate_channels = config['intermediate_channels']
    cell_type = config['cell_type']

    cfm = Cifar10DataModule(batch_size=batch_size)
    model = RegNet(
        rnn_regulated_block, cfm.image_dims[0], intermediate_channels,
        classes=cfm.num_classes, cell_type=cell_type, layers=layers, config=config
    )

    ### Log metric progression
    logger = TensorBoardLogger('tuner_logs', name='regnet_logs')


    #Tune callback
    tune_report = TuneReportCallback({ "loss": "val_loss", "val_accuracy": "val_accuracy"}, on="validation_end")
    tune_report_ckpt = TuneReportCheckpointCallback(
        metrics={ "val_loss": "val_loss", "val_accuracy": "val_accuracy"},
        filename="tune_last_ckpt", on="validation_end"
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs, gpus= num_gpus, logger=logger,
        callbacks=[ tune_report, tune_report_ckpt ]
    )
    trainer.fit(model, cfm)


#======================================= Tuning Functions ============================================

def TuneAsha(train_fn, model:str, num_samples:int=10, num_epochs:int=10, cpus_per_trial:int=1, gpus_per_trial:int=1, data_dir='./tuner'):
    config = {
        "block1": tune.randint(2, 3),
        "block2": tune.randint(2, 5),
        "block3": tune.randint(2, 5),
        "block4": tune.randint(2, 5),
        "cell_type": tune.choice(['gru', 'lstm']),
        "intermediate_channels": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-4, 1e-5),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[ "block1", "block2", "block3", "block4", "lr", "batch_size", "weight_decay"],
        metric_columns=["val_loss", "val_accuracy", "training_iteration"]
    )

    analysis = tune.run(
        tune.with_parameters(
            train_fn, num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_loss", mode="min", config=config,
        num_samples=num_samples, scheduler=scheduler, progress_reporter=reporter, name=f"{model}_asha"
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    exit(0)

def TunePBT(train_fn, model:str, num_samples:int=10, num_epochs:int=10, cpus_per_trial:int=1, gpus_per_trial:int=1, data_dir='./tuner'):
    config = {
        "block1": tune.randint(2, 3),
        "block2": tune.randint(2, 5),
        "block3": tune.randint(2, 5),
        "block4": tune.randint(2, 5),
        "cell_type": tune.choice(['gru', 'lstm']),
        "intermediate_channels": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-4, 1e-5),
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "block1": tune.randint(2, 3),
            "block2": tune.randint(2, 5),
            "block3": tune.randint(2, 5),
            "block4": tune.randint(2, 5),
            "cell_type": ['gru', 'lstm'],
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e-5),
            "batch_size": [32, 64, 128],
            "intermediate_channels": [16, 32, 64],
        }
    )

    reporter = CLIReporter(
        parameter_columns=[ "block1", "block2", "block3", "block4", "lr", "batch_size", "weight_decay"],
        metric_columns=["val_loss", "val_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_fn, num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_loss", mode="min", config=config,
        num_samples=num_samples, scheduler=scheduler, progress_reporter=reporter, name=f"{model}_pbt"
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    exit(0)

#====================================== End Tuning Functions =====================================



if __name__  == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', help='Find hyperparameter values', action='store_true')
    args = parser.parse_args()
    if args.tune:
        TunePBT(
            train_regnet, 'regnet', num_samples=30, num_epochs=2,
            cpus_per_trial=4, gpus_per_trial=1
        )

    cfm = Cifar10DataModule(batch_size=batch_size)
    model = RegNet(rnn_regulated_block, cfm.image_dims[0], 32, cfm.num_classes, 'gru', [2, 2, 2, 2])


    ### Log metric progression
    logger = TensorBoardLogger('logs', name='regnet_logs')

    ### Callbacks
    stop_early = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    last_chkpt_path = 'checkpoints/last.ckpt'
    checkpoint = ModelCheckpoint(
        dirpath= last_chkpt_path, monitor='val_accuracy', mode='max',
        filename='{epoch}-{val_accuracy:.2f}', verbose=True, save_top_k=1
    )


    trainer = Trainer(
        gpus=0, fast_dev_run=True, logger=logger,
        max_epochs=max_epochs, callbacks=[checkpoint],
    )

    trainer.fit(model, cfm)


