import torch
import torch.nn as nn
from clip import clip
import math

representations_dims = {'RN50': 1024,
                        'RN101': 512,
                        'RN50x4': 640,
                        'RN50x16': 768,
                        'ViT-B/32': 512,
                        'ViT-B/16': 512, 
                        'ViT-L/14': 768 }

class LinearWrapper(torch.nn.Module): 
    def __init__(self, in_features, num_classes,initial_weights=None): 
        super(LinearWrapper, self).__init__()
    
        self.classification_head = torch.nn.Linear(in_features, num_classes)
        
        if initial_weights is not None and type(initial_weights) == tuple:
            print('tuple.')
            w, b = initial_weights
            self.classification_head.weight = torch.nn.Parameter(w.clone())
            self.classification_head.bias = torch.nn.Parameter(b.clone())
        else:
            if initial_weights is None:
                initial_weights = torch.zeros_like(self.classification_head.weight)
                torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
            # Note: modified. Initial bug in forgetting to zero bias.
            self.classification_head.bias = torch.nn.Parameter(torch.zeros_like(self.classification_head.bias))

    def forward(self, features): 

        return self.classification_head(features)

class ModelWrapper(torch.nn.Module):
    def __init__(self, backbone, normalize=True):
        super(ModelWrapper, self).__init__()

        self.model, _ = clip.load(backbone)
        in_features = self.model.visual.output_dim
        self.d_out = in_features
        self.normalize = normalize
        self.model.visual.float()

        # Note: modified. Get rid of the language part.
        delattr(self.model, 'transformer')

    def forward(self, images):
        # with torch.no_grad():
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features


def ClipRN50(num_classes=10):
    model =  ModelWrapper('RN50')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 
    

def ClipRN101(num_classes=10):
    model =  ModelWrapper('RN101')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 


def ClipRN50x4(num_classes=10):
    model =  ModelWrapper('RN50x4')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 


def ClipRN50x16(num_classes=10):
    model =  ModelWrapper('RN50x16')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 


def ClipViTB16(num_classes=10):
    model =  ModelWrapper('ViT-B/16')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 

def ClipViTB32(num_classes=10):
    model =  ModelWrapper('ViT-B/32')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 


def ClipViTL14(num_classes=10):
    model =  ModelWrapper('ViT-L/14')
    classifier = LinearWrapper(model.d_out, num_classes)   

    return (model, classifier) 