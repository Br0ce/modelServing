## ---------------------------------------------------------------------------
## Use this Makefile to create .mar files for the according models.
## The resulting .mar files will be placed into the model-store folder.
## The model-store folder will be mapped into the torchserve docker container
## by the docker-compose.yml.
## ---------------------------------------------------------------------------

start:
	torchserve --start --ts-config config.properties

stop:
	torchserve --stop

publish-ner:
	torch-model-archiver \
		--model-name ner \
		--version 1.0 \
		--serialized-file ./model/ner/pytorch_model.bin \
		--extra-files "./model/ner/config.json,./model/ner/special_tokens_map.json,./model/ner/tokenizer.json,./model/ner/tokenizer_config.json,./model/ner/vocab.txt" \
		--handler ./handler/ner_handler.py \
		--export-path ./model-store

register-ner:
	curl -X POST "http://localhost:8081/models?url=ner.mar"
	curl -v -X PUT "http://localhost:8081/models/ner?min_workers=2"

publish-summarization:
	torch-model-archiver \
		--model-name summarization \
		--version 1.0 \
		--serialized-file ./model/summarization/pytorch_model.bin \
		--extra-files "./model/summarization/merges.txt,./model/summarization/config.json,./model/summarization/special_tokens_map.json,./model/summarization/tokenizer.json,./model/summarization/tokenizer_config.json,./model/summarization/vocab.json" \
		--handler ./handler/summarization_handler.py \
		--export-path ./model-store

register-summarization:
		curl -X POST "http://localhost:8081/models?url=summarization.mar"
		curl -v -X PUT "http://localhost:8081/models/summarization?min_workers=2"

publish-sentiment:
	torch-model-archiver \
		--model-name sentiment \
		--version 1.0 \
		--serialized-file ./model/sentiment/pytorch_model.bin \
		--extra-files "./model/sentiment/config.json,./model/sentiment/special_tokens_map.json,./model/sentiment/tokenizer_config.json,./model/sentiment/vocab.txt" \
		--handler ./handler/sentiment_handler.py \
		--export-path ./model-store

register-sentiment:
	curl -X POST "http://localhost:8081/models?url=sentiment.mar"
	curl -v -X PUT "http://localhost:8081/models/sentiment?min_workers=2"
