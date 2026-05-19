SHELL := /bin/bash

GOOGLE_APPLICATION_CREDENTIALS = ${HOME}/.config/gcloud/application_default_credentials.json
IMAGE = 2025q1
ARTIFACT_REPO = us-east1-docker.pkg.dev/contrails-301217/contrail-bench

#===============
# Local testing
#===============

docker-build:
	docker build -t $(IMAGE) .

docker-rebuild:
	docker build --no-cache -t $(IMAGE) .

docker-interactive: docker-build
	docker run -it --rm \
		-v $(shell pwd)/pipelines:/pipelines:ro \
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
