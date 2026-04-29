import React, { useEffect, useState } from 'react';
import { getJob } from '../api';

const POLL_MS = 1500;

export default function ResultDisplay({ jobId, onReset }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    setJob(null);
    setError(null);

    const poll = async () => {
      try {
        const data = await getJob(jobId);
        setJob(data);
        if (data.status === 'done' || data.status === 'failed') return;
        setTimeout(poll, POLL_MS);
      } catch (e) {
        setError(e.message);
      }
    };

    poll();
  }, [jobId]);

  if (error) {
    return (
      <div className="card">
        <p className="error-text">Polling error: {error}</p>
        <button className="btn btn-secondary" onClick={onReset}>Try Again</button>
      </div>
    );
  }

  if (!job || job.status === 'pending' || job.status === 'processing') {
    return (
      <div className="card center">
        <div className="spinner" />
        <p className="hint">{job?.status === 'processing' ? 'Running inference...' : 'Waiting in queue...'}</p>
        <p className="job-id">Job: {jobId}</p>
      </div>
    );
  }

  if (job.status === 'failed') {
    return (
      <div className="card">
        <p className="error-text">Job failed: {job.error}</p>
        <button className="btn btn-secondary" onClick={onReset}>Try Again</button>
      </div>
    );
  }

  const { result } = job;

  return (
    <div className="card">
      <h2>Result</h2>
      <div className="prediction-main">
        <span className="prediction-class">{result.predicted_class}</span>
        <span className="prediction-confidence">{(result.confidence * 100).toFixed(1)}%</span>
      </div>

      <div className="top-k">
        {result.top_k.map((item) => (
          <div key={item.class_index} className="top-k-row">
            <span className="top-k-label">{item.class_name}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(item.probability * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="top-k-pct">{(item.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <p className="job-id">Job: {jobId}</p>
      <button className="btn btn-secondary" onClick={onReset}>Analyze Another</button>
    </div>
  );
}
