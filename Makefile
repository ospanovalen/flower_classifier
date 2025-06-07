MANAGER = poetry run
DEVICE = 'cuda:0'

format:
	${MANAGER} isort flower_classifier
	${MANAGER} black flower_classifier

run-train:
run-inference-pipeline:
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
