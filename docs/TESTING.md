# Testing

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ --cov=app --cov-report=term-missing
```

Coverage target: **70%+** (enforced in CI).

---

## Test Structure

### Unit Tests — `test_predictor.py`

Tests preprocessing functions and the predictor class in isolation. No model checkpoint required.

- `test_compute_odba_output_shape` — ODBA calculation produces correct shape
- `test_ensure_feature_matrix_expands_12_to_14` — 12-column input gets ODBA appended
- `test_ensure_feature_matrix_passes_through_14` — 14-column input unchanged
- `test_transform_for_model_no_tf` — raw feature path (no CWT)
- `test_transform_for_model_with_cwt` — time-frequency transform path
- `test_predictor_not_ready_without_checkpoint` — graceful handling of missing model file
- `test_predictor_raises_runtime_when_not_ready` — correct error raised on inference attempt

### API Tests — `test_api.py`

Tests all HTTP endpoints using FastAPI's `TestClient` with an in-memory SQLite database.

- `test_health` — health endpoint returns 200
- `test_submit_predict_returns_job_id` — POST /predict returns job_id + 202
- `test_submit_predict_enqueues_task` — Celery `.delay()` called with correct args
- `test_get_job_not_found` — returns 404 for unknown job_id
- `test_get_job_pending` — pending job returns correct structure
- `test_get_job_done` — done job returns prediction result
- `test_get_job_failed` — failed job returns error message
- `test_predict_top_k_out_of_range` — top_k=0 returns 422
- `test_predict_top_k_max` — top_k=10 accepted

### Queue Tests — `test_tasks.py`

Tests Celery task functions directly (no broker needed). Uses an in-memory SQLite database.

- `test_task_marks_job_done` — successful inference updates job status to done
- `test_task_marks_job_failed_on_error` — exception updates status to failed with error message
- `test_task_sets_processing_status_before_inference` — job transitions through processing state
- `test_enqueue_delay_called_with_correct_args` — `.delay()` receives correct arguments

### Model Loading Tests — `test_model_loading.py`

Tests the model status checker and related endpoints.

- `test_both_missing` — both files missing → not ready
- `test_checkpoint_missing_metadata_present` — checkpoint missing → not ready
- `test_checkpoint_present_metadata_missing` — metadata missing → not ready
- `test_both_present_and_valid` — both valid → ready, correct class names + window size
- `test_bad_metadata_json` — corrupt JSON → not ready
- `test_health_model_not_ready` — health endpoint reflects not-ready status
- `test_health_model_ready` — health endpoint reflects ready status
- `test_model_status_not_ready` — /model/status returns correct detail
- `test_model_status_ready` — /model/status returns class names and window size

---

## Test Infrastructure

**conftest.py** provides:
- In-memory SQLite database (StaticPool — single connection, no file)
- FastAPI `TestClient` with `get_db` dependency overridden
- Database tables created fresh for each test session

Tests never touch Redis, the real database, or the model checkpoint. All external dependencies are mocked.

---

## CI Enforcement

The GitHub Actions `test-backend` job runs:

```bash
ruff check app tests     # lint
pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=70
```

The pipeline blocks merges if coverage drops below 70%.
