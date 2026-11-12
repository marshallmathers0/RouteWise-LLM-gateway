# Every target is a plain Python command, so Windows users without make can copy it.
PY ?= python

.PHONY: install dev ui test test-all up down eval-cache router-queries collect label train-router eval-router load-test demo-data hf-space mlflow-ui

install:
	$(PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt

dev:            ## gateway on :8000
	$(PY) -m uvicorn routewise.main:app --reload --port 8000

ui:             ## playground + dashboard on :8501 (needs the gateway running)
	$(PY) -m streamlit run dashboard/app.py

test:           ## fast tests (mocked providers)
	$(PY) -m pytest -q -m "not slow"

test-all:       ## including tests that load the real embedding/cross-encoder models
	$(PY) -m pytest -q

up:
	docker compose up --build -d

down:
	docker compose down

eval-cache:     ## Phase 3: semantic-cache ablation -> reports/cache_eval.md
	$(PY) -m eval.build_cache_pairs
	$(PY) -m eval.cache_eval

router-queries: ## Phase 4: build the 1,000-query router dataset
	$(PY) -m eval.router_queries

collect:        ## Phase 4: cheap + strong answers (resumable, rate-limited)
	$(PY) -m eval.router_collect

label:          ## Phase 4: LLM judge + labels -> reports/router_labels.md
	$(PY) -m eval.router_label judge
	$(PY) -m eval.router_label build

train-router:   ## Phase 5: train + select on validation -> artifacts/router.joblib (+ MLflow run)
	$(PY) -m eval.router_train

eval-router:    ## Phase 5: test-set evaluation -> reports/router_eval.md
	$(PY) -m eval.router_eval

load-test:      ## Phase 7: gateway overhead with mocked providers -> reports/load_test.md
	$(PY) -m eval.load_test

demo-data:      ## cold-start demo logs + warm semantic cache (real API calls)
	$(PY) scripts/build_demo_data.py

hf-space:       ## assemble build/hf_space for Hugging Face Spaces
	$(PY) scripts/prepare_hf_space.py

mlflow-ui:      ## browse router training runs at http://localhost:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
