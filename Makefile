.PHONY: test test-all install lint bench bench-real docker clean

install:
	pip install -e ".[all]"
	pip install pytest

test:
	pytest tests/ -v -m "not slow" --tb=short

test-all:
	pytest tests/ -v --tb=short

bench:
	aip-bench run halueval ockbench qa_compression --model dummy

bench-real:
	aip-bench run mmlu gsm8k fever --model hf:distilgpt2 -o results/real_results.json

docker:
	docker build -t aip-bench .
	docker run --rm aip-bench

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build .pytest_cache
