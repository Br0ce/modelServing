import logging

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SentimentHandler(BaseHandler):
    """
    Text classification model for sentiment. This handler takes a text (string)
    as input and returns the predicted sentiment.

    Model can be found at https://huggingface.co/oliverguhr/german-sentiment-bert
    """

    def __init__(self):
        super(SentimentHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")

        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f'Sentiment Model loaded successfully from {model_dir}')
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
        return self.tokenizer(sequence, padding=True,
                              truncation=True,
                              add_special_tokens=True,
                              return_tensors="pt")

    def inference(self, inputs, *args, **kwargs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            inputs : Text Tensor from the pre-process function is passed here
        Returns:
            (list) : It returns a list of the predicted value.
        """
        logger.debug("start inference")
        with torch.no_grad():
            outputs = self.model(**inputs).logits

        return outputs

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        prediction = torch.softmax(inference_output, dim=1).tolist()[0]

        result = dict()
        for i, c in enumerate(["positive", "negative", "neutral"]):
            result[c] = prediction[i]

        return [result]
