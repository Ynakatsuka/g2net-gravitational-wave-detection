import inspect

from .base import LightningModuleBase


class LightningModuleNode2Vec(LightningModuleBase):
    def training_step(self, batch, batch_nb):
        pos_rw, neg_rw = batch[0], batch[1]
        loss = self.model.loss(pos_rw, neg_rw)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        outputs = {}
        return outputs


class LightningModuleLightlySSL(LightningModuleBase):
    calculate_val_loss = False

    def forward(self, x):
        return self.model.backbone(x)

    def training_step(self, batch, batch_nb):
        (x0, x1) = batch[0]
        y0, y1 = self.model.forward(x0, x1)

        loss_args = inspect.getfullargspec(self.hooks.loss_fn).args
        if "epoch" in loss_args:
            loss = self.hooks.loss_fn(y0, y1, epoch=self.current_epoch)
        else:
            loss = self.hooks.loss_fn(y0, y1)

        return {"loss": loss}
