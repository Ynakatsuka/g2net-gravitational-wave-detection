import kvt
import kvt.hooks
import matplotlib.pyplot as plt
import numpy as np
import torch
from kvt.hooks import VisualizationHookBase
from PIL import Image
from torchcam import cams
from torchcam.cams import locate_candidate_layer
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


def make_subset_dataloader(dataloader, indices, batch_size):
    new_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataloader.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        sampler=dataloader.sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        drop_last=dataloader.drop_last,
    )
    return new_dataloader


def hstack_pil_image(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


@kvt.HOOKS.register
class G2NetGradCamVisualizationHook(VisualizationHookBase):
    def __init__(
        self,
        dirpath,
        experiment_name,
        figsize=(40, 20),
        method="GradCAMpp",
        num_plots=7,
        input_shape=(3, 224, 224),
        target_layer=None,
        select_top_predictions=True,
    ):
        suffix = "_top" if select_top_predictions else "_bottom"
        super().__init__(dirpath, experiment_name, figsize, suffix)
        self.method = method
        self.num_plots = num_plots
        self.input_shape = input_shape
        self.target_layer = target_layer
        self.select_top_predictions = select_top_predictions

    def _extract_cam_on_first_batch(self, cam_extractor, model, dataloader):
        inputs, output_cams, images = [], [], []
        batch = iter(dataloader).next()
        for x in batch["x"]:
            out = model(x.unsqueeze(0))
            img = model(x.unsqueeze(0), return_spectrogram=True)
            activation_map = cam_extractor(out.unsqueeze(0).argmax().item(), out)
            inputs.append(x)
            output_cams.append(activation_map)
            images.append(img)
        return inputs, output_cams, images

    def _plot_cams(self, save_path, inputs, cams, images, predictions, targets):
        n_cols = 2
        n_rows = self.num_plots

        plt.clf()
        _, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, tight_layout=True)

        for i in range(n_rows):
            axes[i][0].plot(inputs[i].numpy().T)

            inp = images[i][0].numpy().transpose(1, 2, 0)
            inp = (255 * (inp - inp.min()) / inp.max()).astype("uint8")
            overlay_cam = overlay_mask(
                to_pil_image(inp), to_pil_image(cams[i], mode="F"), alpha=0.5
            )
            original_inputs = to_pil_image(inp)
            to_show = hstack_pil_image(original_inputs, overlay_cam)
            axes[i][1].imshow(to_show)
            axes[i][1].set_title(f"Pred: {predictions[i]}, Target: {targets[i]}")
            axes[i][1].axis("off")

        plt.savefig(save_path)

    def __call__(self, model, dataloader, predictions, targets):
        if self.target_layer is None:
            self.target_layer = "backbone." + locate_candidate_layer(model.backbone)
            print("[Target Layer of CAM] ", self.target_layer)

        cam_extractor = getattr(cams, self.method)(
            model, input_shape=self.input_shape, target_layer=self.target_layer
        )

        # choose images that will be visualized
        assert predictions.shape == targets.shape
        deviation = ((predictions - targets) ** 2).mean(axis=1)
        if self.select_top_predictions:
            indices = np.argsort(deviation)[: self.num_plots]
        else:
            indices = np.argsort(deviation)[-self.num_plots :]

        # create new dataloader
        new_dataloader = make_subset_dataloader(dataloader, indices, self.num_plots)
        top_inputs, top_cams, images = self._extract_cam_on_first_batch(
            cam_extractor, model, new_dataloader
        )
        self._plot_cams(
            self.save_path,
            top_inputs,
            top_cams,
            images,
            predictions[indices],
            targets[indices],
        )

        return self.result
