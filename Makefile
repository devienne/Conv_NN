.PHONY: install pipeline train api dashboard test lint clean

install:
	pip install -e ".[dev]"

pipeline:
	python scripts/run_pipeline.py --config configs/experiment_convnetquake.yaml

train:
	python scripts/train.py --config configs/experiment_convnetquake.yaml

api:
	uvicorn seismic_cnn.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/seismic_cnn/dashboard/app.py

test:
	pytest tests/ -v --cov=seismic_cnn --cov-report=term-missing

lint:
	ruff check src/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
