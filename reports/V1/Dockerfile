# https://cloud.google.com/dataflow/docs/guides/build-container-image#use_a_custom_base_image

ARG BEAM_SDK_VERSION=2.67.0
ARG PYTHON_VERSION=3.12

# need to copy beam from this image later
# https://stackoverflow.com/questions/43473236/docker-build-arg-and-copy
FROM apache/beam_python${PYTHON_VERSION}_sdk:${BEAM_SDK_VERSION} AS beam_sdk

FROM python:${PYTHON_VERSION} AS builder

ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install -U pip
RUN pip install pipenv
RUN mkdir -p /tmp/contrail-bench
COPY Pipfile Pipfile.lock /tmp/contrail-bench/
RUN PIPENV_PIPFILE=/tmp/contrail-bench/Pipfile pipenv install --deploy --system
RUN rm -rf /tmp/contrail-bench

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG PYTHON_VERSION=3.12

COPY --from=beam_sdk /opt/apache/beam /opt/apache/beam
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages

ENTRYPOINT ["/opt/apache/beam/boot"]
