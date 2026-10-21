# https://cloud.google.com/dataflow/docs/guides/build-container-image#use_a_custom_base_image

ARG BEAM_SDK_VERSION=2.76.0
ARG PYTHON_VERSION=3.13

# need to copy beam from this image later
# https://stackoverflow.com/questions/43473236/docker-build-arg-and-copy
FROM apache/beam_python${PYTHON_VERSION}_sdk:${BEAM_SDK_VERSION} AS beam_sdk

FROM python:${PYTHON_VERSION} AS builder

# pycontrails comes from a private fork (see Pipfile), which needs an SSH key at build time
RUN mkdir -p /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install -U pip
RUN pip install pipenv
RUN mkdir -p /tmp/contrail-bench
COPY Pipfile Pipfile.lock /tmp/contrail-bench/
RUN --mount=type=ssh,id=github_ssh_key PIPENV_PIPFILE=/tmp/contrail-bench/Pipfile pipenv install --deploy --system
RUN rm -rf /tmp/contrail-bench

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG PYTHON_VERSION=3.13

# No system libraries are needed: cfgrib's eccodes library comes in the ``eccodeslib`` wheel

COPY --from=beam_sdk /opt/apache/beam /opt/apache/beam
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages

COPY src/ /src/

ENTRYPOINT ["/opt/apache/beam/boot"]
