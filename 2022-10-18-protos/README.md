To compile the generated protobuf files:

```bash
protoc -I=vendor --python_out=. vendor/**/*.proto
```

Requires Protocol Buffers v21 and `mypy-protobuf`.
