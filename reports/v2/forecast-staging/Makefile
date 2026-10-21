SHELL := /bin/bash

GITHUB_SSH_KEY = ${HOME}/.ssh/id_ed25519_gh
GOOGLE_APPLICATION_CREDENTIALS = ${HOME}/.config/gcloud/application_default_credentials.json
IMAGE = v2-forecast-staging
ARTIFACT_REPO = us-east1-docker.pkg.dev/contrails-301217/contrail-bench
RUFF = --config ../../../pyproject.toml --config 'lint.isort.known-first-party=["src"]'

#===============
# Setup
#===============

install:
	pipenv install --dev

lock:
	pipenv lock

#===============
# Quality
#===============

lint:
	pipenv run ruff check $(RUFF) --no-force-exclude src pipelines
	pipenv run ruff format $(RUFF) --no-force-exclude --check src pipelines

format:
	pipenv run ruff check $(RUFF) --no-force-exclude --fix src pipelines
	pipenv run ruff format $(RUFF) --no-force-exclude src pipelines

#===============
# Local testing
#===============

docker-build:
	docker build --ssh github_ssh_key=$(GITHUB_SSH_KEY) -t $(IMAGE) .

docker-rebuild:
	docker build --no-cache --ssh github_ssh_key=$(GITHUB_SSH_KEY) -t $(IMAGE) .

docker-interactive: docker-build
	docker run -it --rm \
		-v $(shell pwd)/src:/src:ro \
		-v $(GOOGLE_APPLICATION_CREDENTIALS):/tmp/keys/application_default_credentials.json:ro \
		-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/application_default_credentials.json \
		--entrypoint /bin/bash \
		$(IMAGE)

#========================
# Deployment to Dataflow
#========================

docker-push: docker-build
	docker tag $(IMAGE) $(ARTIFACT_REPO)/$(IMAGE)
	docker push $(ARTIFACT_REPO)/$(IMAGE)
