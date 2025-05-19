FROM golang:1.24-alpine AS builder

WORKDIR /build

COPY main.go .

RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/hunyuan main.go

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/hunyuan /app/hunyuan

RUN chmod +x /app/hunyuan

EXPOSE 6666

CMD ["/app/hunyuan", "--address", "0.0.0.0", "--port", "6666", "--verify-ssl=false", "--dev", "--workers", "400", "--queue-size", "1000", "--max-concurrent", "400"]
