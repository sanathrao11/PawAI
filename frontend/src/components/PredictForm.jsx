import React, { useState } from 'react';
import { generateDemoWindow, parseCSV, submitPredict } from '../api';

export default function PredictForm({ onJobSubmitted }) {
  const [topK, setTopK] = useState(3);
  const [csvError, setCsvError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(window) {
    setLoading(true);
    try {
      const job = await submitPredict(window, topK);
      onJobSubmitted(job.job_id);
    } catch (e) {
      setCsvError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleFile(e) {
    setCsvError(null);
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const window = parseCSV(ev.target.result);
        submit(window);
      } catch (err) {
        setCsvError(err.message);
      }
    };
    reader.readAsText(file);
  }

  function handleDemo() {
    setCsvError(null);
    submit(generateDemoWindow(200));
  }

  return (
    <div className="card">
      <h2>Analyze Behavior</h2>
      <p className="hint">
        Upload CSV (rows = timesteps, cols = 12 or 14 sensor channels) or use demo data.
      </p>

      <div className="form-row">
        <label>
          Top-K results
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[1, 2, 3, 5].map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="form-actions">
        <label className={`btn btn-primary ${loading ? 'disabled' : ''}`}>
          Upload CSV
          <input type="file" accept=".csv,.txt" onChange={handleFile} hidden disabled={loading} />
        </label>
        <button className="btn btn-secondary" onClick={handleDemo} disabled={loading}>
          Use Demo Data
        </button>
      </div>

      {loading && <p className="hint">Submitting job...</p>}
      {csvError && <p className="error-text">{csvError}</p>}
    </div>
  );
}
