import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SummarizationHandler(BaseHandler):
    """
    Text summarization model. This handler takes a text (string)
    as input and returns a summarization

    Model can be found at https://huggingface.co/facebook/bart-large-cnn
    """

    def __init__(self):
        super(SummarizationHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")

        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f'Summarization Model loaded successfully from {model_dir}')
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
        return self.tokenizer.encode("summarize: " + sequence,
                                     return_tensors="pt",
                                     max_length=512,
                                     truncation=True)

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
            outputs = self.model.generate(inputs,
                                          max_length=150,
                                          min_length=30,
                                          length_penalty=2.0,
                                          num_beams=4,
                                          early_stopping=True)

        return outputs

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        summary = self.tokenizer.decode(inference_output[0],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)

        return [{"summary": summary}]
