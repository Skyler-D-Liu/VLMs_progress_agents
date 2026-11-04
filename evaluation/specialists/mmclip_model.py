"""Mmclip model."""
import torch
from torch import nn
from transformers import CLIPModel


class ImageEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(backbone, local_files_only=True)

    def forward(self, pixels):
        model = self.clip_model
        return model.visual_projection(model.vision_model(pixels).pooler_output)


class TextEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(backbone, local_files_only=True)

    def forward(self, tokens):
        model = self.clip_model
        return model.text_projection(model.text_model(**tokens).pooler_output)


class MMCLIP(nn.Module):
    def __init__(self, backbone, output_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(backbone)
        self.text_encoder = TextEncoder(backbone)
        image_dim = self.image_encoder.clip_model.config.projection_dim
        text_dim = self.text_encoder.clip_model.config.projection_dim
        self.satellite_projector = nn.Sequential(nn.Linear(image_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.text_projector = nn.Sequential(nn.Linear(text_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

    def load_checkpoint(self, state):
        expected = set(self.state_dict())
        omitted = set(state)-expected
        if any(not key.startswith(('ground_projector.', 'location_encoder.')) and key != 'logit_scale' for key in omitted):
            raise ValueError('Checkpoint contains unrecognized MMCLIP parameters')

        self.load_state_dict({key:value for key,value in state.items() if key in expected}, strict=True)

    def forward(self, satellite=None, text=None):
        image = self.satellite_projector(self.image_encoder(satellite)) if satellite is not None else None
        language = self.text_projector(self.text_encoder(text)) if text is not None else None
        return None, image, language, None
