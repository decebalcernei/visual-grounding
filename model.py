"""
Implemetation similar to TransVG architecture, using CLIP's visual and linguistic embeddings
instead of the papers's respective branches.
So we take the CLIP encodings and create a visual-linguistic fusion layer.

The paper's fusion layer is composed by two linear projection layers (one for each modality)
and a visual linguistic transformer (composed by a stack of 6 tranformer encoder layers).

The linear projection layers simply projects the different input dimensions of the two modality
to a embedding with the same channels dimension (cp=256).

To the visual encoder layer we will add a learnable embedding token ([REG]) which will then be used
to predict the actual bbox coordinates.
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import types


class VisualLanguisticTranformer(nn.Module):

    def __init__(self, clip_model):
        super(VisualLanguisticTranformer, self).__init__()

        # modified_part = types.MethodType(new_part, where_to_attach_it_to)
        clip_model.visual.forward = types.MethodType(modified_visual_forward, clip_model.visual)
        clip_model.encode_text = types.MethodType(modified_encode_text, clip_model)
        self.clip_model = clip_model
        self.visual_projection_layer = nn.Linear(2048, 256)
        self.textual_projection_layer = nn.Linear(1024, 256)


    def forward(self, image, text):
        with torch.no_grad():
            # print(image.shape) # (batch_size, 3, 224, 224)
            image_embeds = self.clip_model.visual(image) # (batch_size, 2048, 7, 7) # before modifying the visual (batch_size, 1024)
            text_embeds = self.clip_model.encode_text(text) # (batch_size, 77, 1024)

            image_features_flattened = image_embeds.flatten(start_dim=2, end_dim=-1) # (batch_size, 2048, 49)
            text_features_flatten = text_embeds.permute(0, 2, 1) # (batch_size, 1024, 77)

            projected_visual = self.visual_projection_layer(image_features_flattened.permute(0, 2, 1))
            projected_textual = self.textual_projection_layer(text_features_flatten.permute(0, 2, 1))

            print(projected_visual.permute(0, 2, 1).shape)
            print(projected_textual.permute(0, 2, 1).shape)
            reg_token = torch.zeros(1, 256, 1)
            print(reg_token.shape)
            exit()

        return 0




def modified_visual_forward(self, x: torch.Tensor):
    """
        Open AI official implementation, we removed the last attention pooling layer
        to keep more information
    """
    
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # x = self.attnpool(x) <- removed attention pooling layer

    return x


def modified_encode_text(self, text):
    """
        We removed the last operation that was performing an argmax.
        Now instead of having a single embedding for a sentence,
        we have the embedding of each token of the sentence.
    """

    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    x = x @ self.text_projection

    return x
