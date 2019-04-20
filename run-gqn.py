"""
run-gqn.py

Script to train the a GQN dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
"""
import random
import math
from argparse import ArgumentParser

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from gqn import GenerativeQueryNetwork, partition, Annealer
from dataset import GQN_Dataset

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_params = {
    'rooms_ring_dataset': {'max_m': 5},
    'shepart_metzler': {'max_m': 15}
}

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network')
    parser.add_argument('--n_epochs', type=int, default=7, help='number of epochs run (default: 500)')
    parser.add_argument('--batch_size', type=int, default=1, help='multiple of batch size (default: 1)')
    parser.add_argument('--data_dir', type=str, help='location of data', default="train")
    parser.add_argument('--log_dir', type=str, help='location of logging', default="log")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    parser.add_argument('--dataset', type=str, help='dataset name (default: rooms_ring_camera)', default='rooms_ring_dataset')
    parser.add_argument('--L', type=int, help='number of generative steps (def: 6)', default=6)
    parser.add_argument('--pool', type=bool, help='Whether to use Pool representation', default=False)
    parser.add_argument('--dgf_dim', type=int, help='bottleneck size(default=256)', default=256)
    args = parser.parse_args()
    if not(os.path.exists("main.log")):
        with open("main.log", "w") as f:
            f.write("step,elbo,ll,kl,sigma,mu\n")
    if not(os.path.exists("l2.log")):
        with open("l2.log", "w") as f:
            f.write("step,l2\n")

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, dgf_dim=args.dgf_dim, L=args.L, pool=args.pool).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model

    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-4))

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.5, 2 * 10 ** 5)
    mu_scheme = Annealer(5 * 10 ** (-4), 5 * 10 ** (-5), 1.6 * 10 ** 6)

    if len(os.listdir("./checkpoints/"))>0:
        checkpoint_nr = 0
        checkpoint = torch.load("./checkpoints/checkpoint_model_{}.pth".format(checkpoint_nr))
        ch_optimizer = torch.load("./checkpoints/checkpoint_optimizer_{}.pth".format(checkpoint_nr))

        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(ch_optimizer)
        annealers = torch.load("./checkpoints/checkpoint_annealers_{}.pth".format(checkpoint_nr))
        sigma, mu = annealers
        sigma_scheme = Annealer(sigma['init'], sigma['delta'], sigma['steps'])
        sigma_scheme.s = sigma["s"]
        mu_scheme = Annealer(mu['init'], mu['delta'], mu['steps'])
        mu_scheme.s = mu["s"]
        print("Checkpoint loaded")
        print(mu_scheme.s)
        print(sigma_scheme.s)

    # Load the dataset
    train_dataset = GQN_Dataset(root_dir=args.data_dir)
    valid_dataset = GQN_Dataset(root_dir=args.data_dir, train=False)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    max_m = dataset_params[args.dataset]['max_m']

    def step(engine, batch):
        x, v = batch
        x, v = x.to(device), v.to(device)
        x, v, x_q, v_q = partition(x, v, max_m)

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x, v, x_q, v_q)

        # Log likelihood
        sigma = next(sigma_scheme)
        ll = Normal(x_mu, sigma).log_prob(x_q)

        likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward() #

        optimizer.step()
        optimizer.zero_grad()


        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = mu_scheme.s
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        if not sigma_scheme.s%10:
            with open("main.log", "a") as f:
                output = list(map(str, [sigma_scheme.s, -elbo.item(), -likelihood.item(), kl_divergence.item(), sigma, mu]))
                f.write(",".join(output)+"\n")

        return {"elbo": elbo.item(), "kl": kl_divergence.item(), "sigma": sigma, "mu": mu}

    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ["elbo", "kl", "sigma", "mu"]
    metrics = [RunningAverage(output_transform=lambda x: x[m]).attach(trainer, m) for m in metric_names]
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint("./checkpoints", "checkpoint", save_interval=10000, n_saved=10,
                                         require_empty=False)
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                       'annealers': (sigma_scheme.data, mu_scheme.data)})

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_images(engine):
        with torch.no_grad():
            x, v = engine.state.batch
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v, max_m)

            x_mu, r, _ = model(x, v, x_q, v_q)

            r = r.view(-1, 1, 16, 16)

            # Send to CPU
            x_mu = x_mu.detach().cpu().float()
            r = r.detach().cpu().float()

            writer.add_image("representation", make_grid(r), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(x_mu), engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        with torch.no_grad():
            x, v = next(iter(valid_loader))
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v, max_m)

            # Reconstruction, representation and divergence
            x_mu, _, kl = model(x, v, x_q, v_q)
            x_mu *=255
            x_q *=255

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar("validation/elbo", elbo.item(), engine.state.epoch)
            writer.add_scalar("validation/kl", kl_divergence.item(), engine.state.epoch)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, { 'exception': model })
        else: raise e

    trainer.run(train_loader, args.n_epochs)
    writer.close()