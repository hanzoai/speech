FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY main.py stt.py tts.py transcript.py vad.py ./

# The gate, as a build stage. `docker build` is the ONE thing this lane is
# already known to do, so the test needs no uv on the runner and no bind mount
# — the two ways a step here can fail for reasons that have nothing to do with
# the code. It sits ABOVE the weight bake on purpose: the suite stubs both model
# calls, so it needs no weights, and running it first means a contract break is
# caught before 350MB is downloaded rather than after.
FROM base AS test
COPY bench.py test_speech.py ./
RUN uv sync --frozen --group dev && uv run pytest -q

# Weights bake at BUILD: a deterministic image that boots without the network.
FROM base AS weights
RUN uv run python -c "import stt; [stt.load(m) for m in stt.MODELS]" \
 && curl -fsSLo kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
 && curl -fsSLo voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# The second gate: the same code against the REAL models. The suite above replaces
# faster-whisper with a stand-in so the window arithmetic can be checked exactly,
# and a suite that only ever sees a stand-in cannot tell a working decoder from a
# deleted one. Here kokoro speaks a sentence and whisper has to say it back, so
# the weights, the raw-audio path and the sample scaling are all load-bearing.
#
# It sits above `final` and below nothing: `final` does not depend on it, so an
# untargeted `docker build` still produces the image WITHOUT dev dependencies.
FROM weights AS live
COPY test_speech.py test_live.py ./
RUN uv sync --frozen --group dev && uv run pytest -q

FROM weights AS final
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
