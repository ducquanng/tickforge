# Build stage: compile the Rust extension into a wheel.
FROM rust:1-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir "maturin>=1.7,<2"
WORKDIR /src
COPY . .
RUN /venv/bin/maturin build --release --out /wheels

# Runtime stage: Python only.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl matplotlib websockets && rm -rf /wheels
WORKDIR /work
VOLUME ["/work/data", "/work/results"]
ENTRYPOINT ["tickforge"]
