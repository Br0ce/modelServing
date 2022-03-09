import logging
import os.path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import load_label_mapping

logger = logging.getLogger(__name__)


def wordpiece2word(tokens_labels) -> list:
    results = []
    for token, label in tokens_labels:
        if "##" in token:
            res = results[len(results) - 1]  # could not be first
            res['token'] = res['token'] + token.replace("##", "")
            results[len(results) - 1] = res
        else:
            results.append({'token': token, 'pred': label})
    return results


class NerHandler(BaseHandler):
    """
    Token classification model for Named Entity Recognition. This handler takes a text (string)
    as input and returns the classification for each token found.

    Model can be found at https://huggingface.co/philschmid/gbert-base-germaner
    """

    def __init__(self):
        super(NerHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")

        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        sp_tokens_map = load_label_mapping(os.path.join(model_dir, "special_tokens_map.json"))
        self.special_tokens_map = list(sp_tokens_map.values())

        self.model.to(self.device)
        self.model.eval()

        logger.info(f'NER Model loaded successfully from {model_dir}')
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the request data to an input sequence.

        Args :
            data (list): List of the data from the request input. Only the first entry will be
            processed.
        Returns:
            tensor: Returns the tensor data of the input
        """
        logger.debug("preprocessing data")

        sequence = data[0].get("body")
        return self.tokenizer.encode(sequence, return_tensors="pt")

    def inference(self, inputs, *args, **kwargs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            inputs : Text Tensor from the pre-process function is passed here
        Returns:
            (list) : It returns a list of the predicted value for each token.
        """
        logger.debug("start inference")
        with torch.no_grad():
            outputs = self.model(inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
        tokens_labels = self.zip_tokens_labels_remove_specials(tokens, predictions)

        return wordpiece2word(tokens_labels)

    def zip_tokens_labels_remove_specials(self, tokens, predictions):
        """Return list of token label tuples and remove special tokens.
        """
        return [(token, self.id2label(prediction)) for token, prediction in
                zip(tokens, predictions) if token not in self.special_tokens_map]

    def id2label(self, id: int) -> str:
        return self.model.config.id2label[id]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return [inference_output]
