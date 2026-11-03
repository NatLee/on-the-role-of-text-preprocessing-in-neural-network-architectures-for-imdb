# CPU environment with the exact versions behind the README numbers.
#   docker build -t preproc-imdb .
#   docker run --rm -v "$PWD/data:/work/data" -v "$PWD/runs:/work/runs" preproc-imdb train --per-class 500
FROM python:3.12-slim
WORKDIR /opt/preproc-imdb
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
WORKDIR /work
ENTRYPOINT ["preproc-imdb"]
CMD ["--help"]
