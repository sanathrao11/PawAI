# API Reference

Base URL: `http://pawai-alb-1023454956.eu-north-1.elb.amazonaws.com`

---

## GET /health

Service liveness check.

**Response 200:**
```json
{
  "status": "ok",
  "model_ready": false
}
```

Note: `model_ready` reflects the API container's local filesystem. The model lives in the worker — predictions work regardless of this flag.

---

## GET /model/status

Detailed model file validation.

**Response 200:**
```json
{
  "ready": true,
  "checkpoint_exists": true,
  "metadata_exists": true,
  "metadata_valid": true,
  "checkpoint_path": "/app/best_model.pt",
  "metadata_path": "/app/model_metadata.json",
  "class_names": ["Carrying object", "Drinking", "Eating", "Galloping", "Lying chest",
                  "Pacing", "Panting", "Playing", "Shaking", "Sitting", "Sniffing",
                  "Standing", "Synchronization", "Trotting", "Walking"],
  "window_size": 200,
  "error": null
}
```

---

## POST /predict

Submit a sensor window for asynchronous inference.

**Request body:**
```json
{
  "window": [
    [ax, ay, az, nx, ny, nz, gx, gy, gz, gnx, gny, gnz, odba_back, odba_neck],
    ...
  ],
  "top_k": 3
}
```

- `window`: 2D array of shape `(timesteps, channels)`. Minimum 10 timesteps. Accepts 12 columns (raw IMU, ODBA auto-computed) or 14 columns (with ODBA included).
- `top_k`: number of top predictions to return (1–10, default 3)

**Response 202:**
```json
{
  "job_id": "e3d1da09-fbfb-40fd-8a7a-2eb0a7906b05",
  "status": "pending"
}
```

**Errors:**
- `422` — invalid input (wrong column count, non-numeric values, `top_k` out of range)
- `500` — internal error (Redis unavailable, DB error)

---

## GET /jobs/{job_id}

Poll job status and retrieve result.

**Response 200 (pending/processing):**
```json
{
  "job_id": "e3d1da09-fbfb-40fd-8a7a-2eb0a7906b05",
  "status": "processing",
  "result": null,
  "error": null
}
```

**Response 200 (done):**
```json
{
  "job_id": "e3d1da09-fbfb-40fd-8a7a-2eb0a7906b05",
  "status": "done",
  "result": {
    "predicted_index": 14,
    "predicted_class": "Walking",
    "confidence": 0.7004,
    "top_k": [
      { "class_index": 14, "class_name": "Walking", "probability": 0.7004 },
      { "class_index": 13, "class_name": "Trotting", "probability": 0.1823 },
      { "class_index": 5,  "class_name": "Pacing",   "probability": 0.0712 }
    ],
    "window_size": 200,
    "model_ready": true
  },
  "error": null
}
```

**Response 200 (failed):**
```json
{
  "job_id": "...",
  "status": "failed",
  "result": null,
  "error": "Model checkpoint not found at /app/best_model.pt."
}
```

**Response 404** — job not found

---

## Job Status Lifecycle

```
pending → processing → done
                    → failed
```

Typical latency from submission to result: 1–3 seconds on CPU inference.

---

## CSV Format

For file upload via the frontend:

- One row per timestep
- Comma-separated values, no header
- 12 columns (raw IMU) or 14 columns (with ODBA):

```
ABack_x, ABack_y, ABack_z, ANeck_x, ANeck_y, ANeck_z,
GBack_x, GBack_y, GBack_z, GNeck_x, GNeck_y, GNeck_z
[, ODBA_ABack, ODBA_ANeck]
```

Minimum 10 rows. Recommended: 200 rows (2 seconds at 100 Hz).
