import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from typing import Union, Optional


class CNNCoder(nn.Module):
    """
    A simple convolutional encoder that expects inputs of size
    (batch, in_channels, target_h, target_w) and outputs a latent
    vector of size `latent_dim`.
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        target_h: int = 84,
        target_w: int = 84,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1) 3×84×84 → 16×42×42
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 2) 16×42×42 → 32×21×21
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 3) 32×21×21 → 64×11×11  (21→11 with stride2+ceil)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # flatten size = 64 × ceil(84/8) × ceil(84/8)
            #           = 64 × 11 × 11 = 7744
            nn.Linear(64 * ((target_h + 7) // 8) * ((target_w + 7) // 8), latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Preprocessor:
    def encode(self, image) -> torch.Tensor:
        return image


class PreprocessingCNN(Preprocessor):
    """
    Wrapper around CNNCoder that:
     1) Resizes any incoming (H,W,C) or (C,H,W) image to (target_h, target_w)
     2) Normalizes uint8→[0,1]
     3) Runs through the encoder to produce a latent vector
    """

    def __init__(
        self,
        device: Union[torch.device, str, None] = None,
        latent_dim: int = 128,
        input_channels: int = 3,
        target_size: tuple[int, int] = (84, 84),
        model_path: Optional[str] = None,
    ):
        """
        Args:
            device:      torch.device("cpu") or torch.device("cuda")
            latent_dim:  length of output vector
            input_channels: usually 3 for RGB
            target_size: (height, width) to resize EVERY image to
            model_path:  optional path to pretrained encoder weights
        """
        self.device = device
        self.target_h, self.target_w = target_size

        if model_path is None:
            # Default: use custom CNNCoder
            self.encoder = CNNCoder(
                input_channels=input_channels,
                latent_dim=latent_dim,
                target_h=self.target_h,
                target_w=self.target_w,
            ).to(device)
        else:
            # Load checkpoint
            ckpt = torch.load(model_path, map_location=device)
            # If checkpoint specifies a Cadene model and state_dict
            if isinstance(ckpt, dict) and "model_name" in ckpt and "state_dict" in ckpt:
                name = ckpt["model_name"]
                # Instantiate Cadene pretrained model
                model = pretrainedmodels.__dict__[name](
                    num_classes=1000, pretrained="imagenet"
                )
                # Strip off final classification layer
                modules = list(model.children())[:-1]
                feat = nn.Sequential(*modules)
                # Build projection head
                in_features = model.last_linear.in_features
                proj = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(in_features, latent_dim),
                )
                # Full encoder pipeline
                self.encoder = nn.Sequential(feat, proj).to(device)
                # Load adapter weights
                self.encoder.load_state_dict(ckpt["state_dict"])
            else:
                # Fallback: assume ckpt is CNNCoder state_dict
                self.encoder = CNNCoder(
                    input_channels=input_channels,
                    latent_dim=latent_dim,
                    target_h=self.target_h,
                    target_w=self.target_w,
                ).to(device)
                self.encoder.load_state_dict(ckpt)

        self.encoder.eval()

    @torch.no_grad()
    def encode(self, image) -> torch.Tensor:
        """
        image: np.ndarray or torch.Tensor, shape (H,W,C) or (C,H,W), dtype uint8 or float
        returns: FloatTensor of shape (latent_dim,) on self.device
        """
        # 1) to torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        image = image.float()

        # 2) permute if needed → (C,H,W)
        if image.ndim == 3 and image.shape[-1] in {1, 3}:
            image = image.permute(2, 0, 1)

        # 3) normalize 0–255 → [0,1]
        image = image.div(255.0)

        # 4) resize to (target_h, target_w)
        #    expects input shape (C, H, W) → add batch dim → (1,C,H,W)
        image = image.unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )
        image = image.squeeze(0)  # back to (C, H, W)

        # 5) run through encoder → (1, latent_dim) → squeeze → (latent_dim,)
        x = image.unsqueeze(0).to(self.device)
        z = self.encoder(x).squeeze(0)

        return z


from skimage import color
from skimage.transform import rescale


class GrayscaleAndFlattenPreprocessing(Preprocessor):
    def __init__(self, device):
        self.device = device
        pass

    def encode(self, image) -> torch.Tensor:
        rescaled = image# / 255
        grayscale_img = color.rgb2gray(rescaled)
        # print("gray:", grayscale_img)
        torch_img = (torch.from_numpy(grayscale_img) - 0.5) * 2
        # print("gray after rescale:", trch_img)
        return torch_img.flatten().float().to(self.device)


class RescalePreprocessing(Preprocessor):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale

    def encode(self, image):
        scaled = rescale(image, self.scale, channel_axis=-1)
        return scaled


class SequentialPreprocessing(Preprocessor):
    def __init__(self, transforms: list[Preprocessor]) -> None:
        super().__init__()
        self.transforms = transforms

    def encode(self, image):
        x = image
        for preprocessor in self.transforms:
            x = preprocessor.encode(x)

        return x
