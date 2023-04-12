import torch
import os
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from dataset import get_dataloaders
from stain_regression_net import StainRegressionNet


def train(run_name, dataset_dir, nr_epochs, device_nr, batch_size, lr, wd, exp_dir, wandb_key, test_mode=True):

    # set device
    device = torch.device('cuda:{}'.format(device_nr) if torch.cuda.is_available() else 'cpu')

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(dataset_dir=dataset_dir, batch_size=batch_size)

    # get baseline model
    net = StainRegressionNet(lr=lr, wd=wd).to(device)

    # log with weights and biases
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb_logger = WandbLogger(project="HE2P53 stain regression", name=run_name, save_dir=exp_dir,
                               settings=wandb.Settings(start_method='fork'))

    wandb_logger.log_hyperparams({
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": nr_epochs,
        'weight decay': wd
    })

    # save top 5 lowest loss
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(exp_dir, run_name),
                                          filename='{epoch:02d}_{step:03d}_{val loss:.2f}',
                                          monitor='val loss', mode='min', save_top_k=5)

    # 100 epochs no improvement => stop training
    early_stop_callback = EarlyStopping(monitor="val loss", min_delta=0.00, patience=100, verbose=False, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # define trainer
    trainer = Trainer(logger=wandb_logger,
                      accelerator='gpu',
                      callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
                      devices=[device_nr],
                      max_epochs=nr_epochs,
                      log_every_n_steps=1)

    # train the model
    trainer.fit(net, train_loader, val_loader)

    # test the model
    if test_mode:
        print('Testing model: {}'.format(trainer.checkpoint_callback.best_model_path))

        # get results on the test set
        trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=True)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="name of this run")
    parser.add_argument("--dataset_dir", type=str, default='/data/archief/AMC-data/Barrett/temp/patch_dataset_s0.25_512',
                        help="directory where to patch dataset is stored")
    parser.add_argument("--nr_epochs", type=int, default=250, help="the number of epochs")
    parser.add_argument("--device_nr", type=int, default=0, help="which of the gpus to use")
    parser.add_argument("--batch_size", type=int, default=128, help="the size of mini batches")
    parser.add_argument("--lr", type=int, default=1e-3, help="the learning rate")
    parser.add_argument("--wd", type=int, default=1e-5, help="weight decay (L2)")
    parser.add_argument("--exp_dir", type=str, default='/data/archief/AMC-data/Barrett/experiments/he2p53/stain_regression/',
                        help="dir where to store experiment results")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--test", type=bool, help="whether to also test", default=False)
    args = parser.parse_args()
    train(args.run_name, args.dataset_dir, args.nr_epochs, args.device_nr, args.batch_size, args.lr, args.wd, args.exp_dir, args.wandb_key, args.test)