# specify python version that we want to install
FROM python:3.9 as base

FROM base as builder

# copy all the requirements/dependency then install in linux terminal
COPY ./requirements.txt ./install.sh ./
RUN ./install.sh && python -m venv /opt/venv

# setup venv as path
ENV PATH="opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

FROM base

# automatically update everytime re-create/re-build the 'image'
RUN apt-get update \
    && apt-get -y install procps

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/apps/project