ARG JLVERSION=1.9.4
ARG PYVERSION=3.11.6
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}
FROM jupyter/base-notebook

WORKDIR /app
COPY . /app

EXPOSE 8888

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

# Install IPython and other useful libraries:
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Julia pre-requisites:
RUN python3 -c 'import pysr'

# Run jupyter
CMD ["jupyter", "notebook"]

