from transformers import DistilBertForSequenceClassification, DistilBertModel
import torch.nn as nn

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x



class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


def initialize_bert_based_model(net, num_classes):

	if net == 'distilbert-base-uncased':
		model = DistilBertClassifier.from_pretrained(
			net,
			num_labels=num_classes)
		d_features = getattr(model, "classifier").in_features
  
		model.classifier = Identity(d_features)
		model.d_out = d_features
	else:
		raise ValueError(f'Model: {net} not recognized.')
	return model