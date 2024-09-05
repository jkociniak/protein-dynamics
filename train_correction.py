import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
import torch

from torch.utils.data import random_split, DataLoader
from src.losses import Loss


class LitManifoldMetricCorrector(pl.LightningModule):
    def __init__(self, corrected_manifold_cfg,
                       encoder_cfg,
                       loss_cfg,
                       optimizer_cfg,
                       scheduler_cfg=None,
                       decoder_cfg=None):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.encoder = instantiate(encoder_cfg)
        self.decoder = None
        if decoder_cfg is None:
            assert loss_cfg['weights']['reconstruction'] == 0., 'If no decoder is provided, reconstruction loss must be 0'
        else:
            self.decoder = instantiate(decoder_cfg)
        self.manifold = instantiate(corrected_manifold_cfg,
                                    correction_encoder=self.encoder,
                                    correction_decoder=self.decoder)
        self.loss = Loss(**loss_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg


    def on_train_epoch_end(self):
        self.logger.experiment.add_scalar('train/epoch', self.current_epoch, self.global_step)

    def training_step(self, batch, batch_idx):
        encoder_opt = self.optimizers()
        encoder_opt.zero_grad()

        if len(batch.shape) == 2:  # we have a single trajectory
            batch = batch[None]

        assert len(batch.shape) == 3, "We expect batch of shape (B, N, D), where B is number of batches, N is the trajectory and D is the input dimension"
        assert batch.shape[0] == 1, "Batch must contain only a single trajectory"
        assert batch.shape[2] == self.manifold.base_manifold.d, "Batch must have the same dimension as the manifold"

        losses = self.loss(self.manifold, batch)
        assert len(losses) > 0, 'Losses must be provided'
        total_loss = 0.
        for name, val in losses.items():
            self.logger.experiment.add_scalar(f'train/{name}_loss', val, self.global_step)
            total_loss += self.loss.weights[name] * val
        self.logger.experiment.add_scalar('train/loss', total_loss, self.global_step)
        self.manual_backward(total_loss)
        #plot_grad_flow(self.named_parameters())

        encoder_opt.step()

        sch1 = self.lr_schedulers()
        sch1.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        return {}

    def test_step(self, batch, batch_idx):
        return {}

    def configure_optimizers(self):
        if self.decoder is not None:
            all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            all_params = self.encoder.parameters()
        optimizer = instantiate(self.optimizer_cfg, params=all_params)
        print(optimizer)
        if self.scheduler_cfg is None:
            scheduler_obj = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.)
        else:
            scheduler_obj = instantiate(self.scheduler_cfg, optimizer=optimizer)

        scheduler = {
            "scheduler": scheduler_obj,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
            "strict": True,
            "name": 'encoder_lr',
        }

        return [optimizer], [scheduler]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # validation
    assert cfg.corrected_manifold.base_manifold_params.d == cfg.encoder.in_features, 'Encoder and manifold dimensions must match'
    tp = cfg.training_params

    dataset = instantiate(cfg.dataset)
    print('cfg dataset:', cfg.dataset)
    print('dataset:', dataset)
    print('dataset length:', len(dataset))
    dataset_lengths = [1., 0., 0.]
    train_dataset, val_dataset, test_dataset = random_split(dataset, dataset_lengths,
                                                            generator=torch.Generator().manual_seed(cfg.dataset.seed))

    train_loader = DataLoader(train_dataset, batch_size=tp.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=tp.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=tp.batch_size)

    print('full dataset length:', len(dataset))
    print('train dataset length:', len(train_dataset))
    print('val dataset length:', len(val_dataset))
    print('test dataset length:', len(test_dataset))
    print('batch size:', tp.batch_size)

    tp = cfg['training_params']
    pl.seed_everything(tp['seed'])

    model = LitManifoldMetricCorrector(cfg['corrected_manifold'],
                                       cfg['encoder'],
                                       cfg['loss'],
                                       cfg['optimizer'],
                                       decoder_cfg=cfg['decoder'],
                                       scheduler_cfg=cfg['scheduler'])

    print('Plotter settings:')
    print(cfg['plotter'])
    default_callbacks = [pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval='step'),
                         pl.callbacks.ModelCheckpoint(monitor='distance_matrix_loss', mode='min', save_top_k=10)]

    custom_callbacks = [instantiate(cfg['plotter'], dataset=dataset)]  # must be done on CPU
    callbacks = default_callbacks + custom_callbacks

    hydra_cfg = HydraConfig.get()
    encoder_name = hydra_cfg.runtime.choices.encoder
    dataset_name = hydra_cfg.runtime.choices.dataset
    name = tp['name'] + '/' + encoder_name + '_' + dataset_name
    logger = pl.loggers.TensorBoardLogger(save_dir=tp['log_dir'], name=name)

    training_params = dict(
        max_epochs=tp['max_epochs'],
        accelerator=tp['accelerator'],
        devices=tp['devices'],
        default_root_dir=tp['trainer_root_dir'],
        logger=logger,
        log_every_n_steps=1,  # we use 1 batch so we want to log at every batch
        callbacks=callbacks,
        limit_val_batches=0.,
    )

    trainer = pl.Trainer(**training_params)

    trainer.fit(model, train_loader, val_loader, ckpt_path=tp.ckpt_path)
    trainer.test(model, ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    my_app()
