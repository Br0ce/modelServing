# ModelServing

#### serve models with [pytorch/serve](https://github.com/pytorch/serve)

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)

---

The provided handlers in `./handler` make use of pretrained models from Hugging Face and can be
downloaded from there.

The downloaded models are placed in `./model`.

* The [NER model](https://huggingface.co/philschmid/gbert-base-germaner) is originally provided as
  tensorflow version. In order to use the pytorch version in `ner_handler.py`, the model is been
  cast.

* The [summarization model](https://huggingface.co/facebook/bart-large-cnn) is extended with its
  tokenizer.

* The [sentiment model](https://huggingface.co/oliverguhr/german-sentiment-bert) is just as
  provided.

Use the `Makefile` to create .mar files and place them in `./model-store`, as required by
pytorch/serve.

```bash
$ make publish-ner
$ make publish-summarization
$ make publish-sentiment
```

Start serving with

```bash
$ make start
```

or build from the `Dockerfile` and serve with

```bash
$ docker run -p 8080:8080 \
-v path-to-model-store:/home/model-server/model-store \
-v path-to-config:/home/model-server/config \
modelserving:1.0
```

Inference with

```bash
$ curl --request POST 'http://localhost:8080/predictions/ner' \
--header 'Content-Type: text/plain' \
--data-raw 'Tom Ford fliegt mit Air Malta nach New York'
```


