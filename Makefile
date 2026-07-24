.PHONY: export train test lint serve build run

export:
	python -m src.export

train:
	python -m src.train

test:
	pytest tests/ -v

lint:
	ruff check .

serve:
	uvicorn api.main:app --reload

build:
	podman build -t mnist-service .

run:
	podman run -p 8000:8000 mnist-service
