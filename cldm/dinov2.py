from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import conv_nd, zero_module

class DinoV2Model(nn.Module):
    def __init__(self, model_name, size):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.size = size

    def forward(self, x):
        # last hidden state of the model is reshaped (b, n, c) -> (b, h, w, c)
        # and resized to the size of the latent space
        out = self.model(x)
        patch_tokens = out.last_hidden_state[:, 1:, :]

        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_grid = patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2)

        return F.interpolate(patch_grid, size=self.size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    import torch
    from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential

    img_in = torch.randn(1, 3, 512, 512)

    input_hint_block = TimestepEmbedSequential(
        DinoV2Model(),
        nn.SiLU(),
        zero_module(conv_nd(dims=2, in_channels=384, out_channels=320, kernel_size=1)),
    )

    print(input_hint_block(img_in, None, None).shape)
