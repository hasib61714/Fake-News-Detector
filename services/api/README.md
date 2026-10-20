# services/api: FastAPI backend

See the [root README](../../README.md) for setup, configuration and the endpoint reference.

- `app/main.py`: application factory (`create_app`), CORS, routers, startup model load.
- `app/inference/model_service.py`: the explicit model interface. `load()` / `refresh()` / `predict()` / `health()`. It loads through `fakenews_ml.artifacts.load_artifact`, which verifies the checksum, preprocessing version and labels. States: `ready`, `not_trained` (no artifact), `error` (artifact unreadable or failed verification). `predict()` returns `uncertain` when confidence is below the artifact's validation-derived threshold. It never fabricates a prediction. Files are re-checked on each request, so a newly trained model is picked up without a restart.
- `app/demo/`: demo mode, deliberately independent of `app.inference`.
- `app/schemas.py`: typed request/response models (mirrored by `apps/web/src/lib/api.ts`).
- `tests/`: pytest. Model-dependent tests train a tiny model with the real `fakenews_ml` code on an in-code corpus and load the real artifact.

```bash
make api          # from the repository root: uvicorn with auto-reload on :8000
make test-api
```
