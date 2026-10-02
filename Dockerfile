# speech — one Go binary over sherpa-onnx, and ffmpeg. No weights: a pod
# fetches them from Hanzo S3 at boot (weights.go), so this image is the code and
# nothing else, and it builds the same for amd64 and arm64.

FROM golang:1.26-bookworm AS build
# ffmpeg is a runtime dependency of the service AND of its suite: the HTTP
# tests encode and decode real containers through it, and skip without it.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go weights.sum ./
# The gate. The suite replaces every model, so it needs no weights, and a red
# suite fails the build before anything is published. The live tests skip here
# (no weights in a build) and say so; they run where the weights are.
RUN go vet ./... && go test -count=1 ./...
# cgo: sherpa-onnx is C++ behind its C API. The shared libraries ship inside the
# Go module, one directory per architecture; they are copied out beside the
# binary because the rpath the module bakes in points into this stage.
RUN CGO_ENABLED=1 go build -trimpath -ldflags='-s -w' -o /out/speech . \
 && mkdir -p /out/lib \
 && cp "$(go env GOMODCACHE)"/github.com/k2-fsa/sherpa-onnx-go-linux@*/lib/"$(uname -m)"-unknown-linux-gnu/*.so /out/lib/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/lib/ /usr/local/lib/
RUN ldconfig
COPY --from=build /out/speech /usr/local/bin/speech
ENV MODELS=/models PORT=8000
EXPOSE 8000
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/speech"]
