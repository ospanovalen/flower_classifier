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

convert-to-tensorrt:
	@echo "Usage: make convert-to-tensorrt input=<onnx_path> output=<trt_path> [batch=1] [precision=fp16]"
	@echo "Example: make convert-to-tensorrt input=models/model.onnx output=models/model.trt batch=4 precision=fp16"

convert-to-tensorrt-run:
	./scripts/convert_to_tensorrt.sh $(input) $(output) $(batch) $(precision)

run-tensorrt-inference:
	${MANAGER} python -m flower_classifier.production.tensorrt_inference \
		--engine-path $(engine) \
		--image-path $(image) \
		$(if $(output),--output-file $(output),) \
		$(if $(benchmark),--benchmark,) \
		$(if $(iterations),--iterations $(iterations),)

start-mlflow-server:
	${MANAGER} python -m flower_classifier.serving.mlflow_server start \
		--model-uri $(model) \
		--host $(host) \
		--port $(port) \
		--workers $(workers)

mlflow-server-predict:
	${MANAGER} python -m flower_classifier.serving.mlflow_server predict \
		--server-url $(server) \
		--image-path $(image) \
		$(if $(output),--output-file $(output),)

mlflow-server-status:
	${MANAGER} python -m flower_classifier.serving.mlflow_server status \
		--server-url $(server)

start-api-server:
	${MANAGER} python -m flower_classifier.serving.run_server \
		--model-path $(model) \
		--host $(host) \
		--port $(port) \
		--device $(device) \
		--workers $(workers)

ddvc-dataset:
	dvc pull

pre-commit-install:
	${MANAGER} pre-commit install

test:
	${MANAGER} pytest tests/

mlflow-ui:
	${MANAGER} mlflow ui --host 127.0.0.1 --port 8080

.PHONY: format run-train run-inference-pipeline run-inference convert-to-onnx convert-to-tensorrt convert-to-tensorrt-run run-tensorrt-inference start-mlflow-server mlflow-server-predict mlflow-server-status start-api-server ddvc-dataset pre-commit-install test mlflow-ui
