from .base import LightningModuleBase
from .sam import disable_bn, enable_bn


class LightningModuleMLM(LightningModuleBase):
    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        hooks=None,
        dataloaders=None,
        max_epochs=None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hooks = hooks
        self.dataloaders = dataloaders
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        val_loss = self.model(**batch).loss
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_outputs = {}
        gathered_outputs = self.gather_outputs(outputs)

        for key, value in gathered_outputs.items():
            if key not in ("y", "y_hat"):
                avg_outputs[key] = value.mean()

        if not self.trainer.running_sanity_check:
            self.log_dict(avg_outputs, on_epoch=True, prog_bar=True, logger=True)

        return avg_outputs
