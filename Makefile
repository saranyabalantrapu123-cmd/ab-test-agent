PYTHON ?= python
PIP ?= pip install --break-system-packages

.PHONY: setup data demo test eval app clean

setup:  ## install dependencies
	$(PIP) -r requirements.txt

data:  ## generate the synthetic experiments + planted ground truth
	$(PYTHON) scripts/generate_experiments.py

demo:  ## analyse every experiment; memos + charts land in reports/
	@test -f data/experiments.csv || $(MAKE) data
	$(PYTHON) -m abagent.cli demo

test:  ## offline test suite (generates its own data in a temp dir)
	$(PYTHON) -m pytest -q

eval:  ## decisions vs planted ground truth, plus a 20-seed robustness sweep
	@test -f data/experiments.csv || $(MAKE) data
	$(PYTHON) -m abagent.cli eval --seeds 20

app:  ## Streamlit UI
	@test -f data/experiments.csv || $(MAKE) data
	streamlit run app.py

clean:
	rm -rf data .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
