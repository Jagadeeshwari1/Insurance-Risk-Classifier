.PHONY: install data train serve test lint drift clean

install:        ## Install all dependencies
	pip install -r requirements.txt

data:           ## Generate synthetic insurance dataset
	python -m src.data.make_dataset --n 20000

train:          ## Train + evaluate + write model bundle, metrics, model card
	python -m src.models.train

serve:          ## Run the prediction API (http://localhost:8000/docs)
	uvicorn api.main:app --reload --port 8000

test:           ## Run the test suite
	pytest -q

lint:           ## Lint with ruff
	ruff check .

drift:          ## Example: PSI drift report (pass CURRENT=path/to.csv)
	python -m src.monitoring.drift --current $(CURRENT)

clean:          ## Remove generated artifacts
	rm -rf models/*.joblib models/metrics.json models/MODEL_CARD.md mlruns __pycache__
