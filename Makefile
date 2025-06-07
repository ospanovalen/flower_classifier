MANAGER = poetry run
DEVICE = 'cuda:0'

format:
	${MANAGER} isort flower_classifier
	${MANAGER} black flower_classifier

run-train:
	${MANAGER} python -m flower_classifier.training.train

run-inference-pipeline:
	@echo "Usage: make run-inference model=<path> input=<path> output=<path>"
	@echo "Example: make run-inference model=models/best_model.ckpt input=data/test output=predictions.json"

run-inference:
	${MANAGER} python -m flower_classifier.inference.batch_predict \
		--model-path $(model) \
		--input-dir $(input) \
		--output-file $(output)

convert-to-onnx:
	${MANAGER} python -m flower_classifier.production.convert_to_onnx \
		--checkpoint-path $(model) \
		--output-path $(output)

dd:
	@dvc remote modify multimodal_embeddings --local access_key_id $(DVC_ACCESS_KEY_ID)
	@dvc remote modify multimodal_embeddings --local secret_access_key $(DVC_SECRET_ACCESS_KEY)
	@dvc config core.no_scm true

ddvc-model: dd
	dvc pull

ddvc-dataset: dd
	dvc pull

pre-commit-install:
	${MANAGER} pre-commit install

test:
	${MANAGER} pytest tests/

mlflow-ui:
	${MANAGER} mlflow ui --host 127.0.0.1 --port 8080

.PHONY: format run-train run-inference-pipeline run-inference convert-to-onnx dd ddvc-model ddvc-dataset pre-commit-install test mlflow-ui
