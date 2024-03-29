import argparse
from dataset import VirtualStainDataset
import torch
import torch.nn as nn
from cbr import CBR5
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import torch.optim as optim
import numpy as np
import wandb
import os
from tqdm import tqdm
from utils import plot_hed_space
from scipy.stats import pearsonr


def train(run_name, nr_epochs, batch_size, lr, exp_dir, wandb_key):
    """ Training procedure for regression of p53 staining from HE patches.
    """

    # load dataset
    dataset_dir = '/data/archief/AMC-data/Barrett/temp/patch_dataset_s0.25_512'
    dataset = VirtualStainDataset(dataset_dir)
    print('Dataset contains: {} pairs.'.format(dataset.__len__()))

    # split into train, val, test: 0.8, 0.1, 0.1
    proportions = [.8, .1, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    train_set, val_set, test_set = random_split(dataset, lengths)
    print('Train pairs: {}'.format(train_set.__len__()))
    print('Val pairs: {}'.format(val_set.__len__()))
    print('Test pairs: {}'.format(test_set.__len__()))

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, drop_last=True, shuffle=True)

    # initialize UNet, input 3 channel (=HE), output 3 channel (=p53)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CBR5().to(device)
    summary(net, input_size=(3, 512, 512), batch_size=batch_size)

    # define loss and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # log with weights and biases
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(project="HE2P53 stain regression", dir=exp_dir)
    wandb.run.name = run_name
    print('')

    lowest_loss = float('inf')

    for epoch in range(nr_epochs):

        # loss and pcc
        train_loss = 0.0
        val_loss = 0.0
        train_pcc = 0.0
        val_pcc = 0.0

        for x, y, d in tqdm(train_loader, desc='Training'):
            x = x.to(torch.float).to(device)
            y = y.to(torch.float).to(device)
            d = d.to(torch.float).to(device)

            # clear gradients
            optimizer.zero_grad()

            # forward batch
            y_pred = net.forward(x).squeeze()

            # compute loss & update
            loss = loss_function(y_pred, d)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pcc += pearsonr(d.cpu().detach().numpy(), y_pred.cpu().detach().numpy())[0]

        # validate
        with torch.no_grad():
            for x, y, d in tqdm(val_loader, desc='Validating'):
                x = x.to(torch.float).to(device)
                y = y.to(torch.float).to(device)
                d = d.to(torch.float).to(device)

                # forward batch
                y_pred = net.forward(x).squeeze()

                # compute loss & update
                loss = loss_function(y_pred, d)
                val_loss += loss.item()
                val_pcc += pearsonr(d.cpu().detach().numpy(), y_pred.cpu().detach().numpy())[0]

                # save one batch of validation examples for plotting
                he_example = x.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.uint8)[0]
                p53_example = y.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.uint8)[0]
                dab_true = float(d.cpu().detach().numpy()[0])
                dab_pred = float(y_pred.cpu().detach().numpy()[0])

        # log & print stats
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_pcc = train_pcc / len(train_loader)
        avg_val_pcc = val_pcc / len(val_loader)
        print('Epoch {}, train loss: {:.3f}, val loss: {:.3f}, train pcc: {:.3f}, val pcc: {:.3f}'.format(epoch,
                                                                                                          avg_train_loss,
                                                                                                          avg_val_loss,
                                                                                                          avg_train_pcc,
                                                                                                          avg_val_pcc))

        # plot predictions on the validation set
        os.makedirs(os.path.join(exp_dir, 'val_predictions'), exist_ok=True)
        pred_save_path = os.path.join(exp_dir, 'val_predictions', 'predictions_epoch_{}.png'.format(epoch + 1))
        plot_hed_space(he_example, p53_example, dab_true, dab_pred, save_path=pred_save_path)

        wandb.log({'epoch': epoch + 1,
                   'train loss': avg_train_loss,
                   'val loss': avg_val_loss,
                   'train pcc': avg_train_pcc,
                   'val pcc': avg_val_pcc,
                   'prediction': wandb.Image(pred_save_path)})

        # safe lowest loss
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)

        if val_loss < lowest_loss:
            lowest_loss = val_loss
            save_dir = os.path.join(exp_dir, 'checkpoints', 'best_model_loss_{}.pt'.format(lowest_loss))
            torch.save(net.state_dict(), save_dir)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/he2p53/stain_regression/',
                        help="output directory for this experiment")
    parser.add_argument("--nr_epochs", type=int, default=250, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="the size of mini batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial the learning rate")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")

    args = parser.parse_args()
    train(run_name=args.run_name,
          exp_dir=args.exp_dir,
          nr_epochs=args.nr_epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          wandb_key=args.wandb_key)
