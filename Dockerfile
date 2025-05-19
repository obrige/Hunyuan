FROM golang:1.22 AS builder

WORKDIR /app

COPY go.mod ./
COPY go.sum ./  

RUN go mod download || true

COPY main.go ./

RUN CGO_ENABLED=0 go build -o myapp .

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/myapp .

EXPOSE 6666

CMD ["./myapp"]
