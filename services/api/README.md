# services/api: FastAPI backend

See the [root README](../../README.md) for setup, configuration and the endpoint reference.

- `app/main.py`: application factory (`create_app`), CORS, routers, startup model load.
- `app/inference/model_service.py`: the explicit model interface. `load()` / `refresh()` / `predict()` / `health()`. States: `ready`, `not_trained` (no artifact), `error` (artifact unreadable or wrong labels). It never fabricates a prediction.
- `app/demo/`: demo mode, deliberately independent of `app.inference`.
- `app/schemas.py`: typed request/response models (mirrored by `apps/web/src/lib/api.ts`).
- `tests/`: pytest. Model-dependent tests fit a tiny toy scikit-learn pipeline in a temp directory.

```bash
make api          # from the repository root: uvicorn with auto-reload on :8000
make test-api
```
