import React, { useEffect, useState } from 'react';
import { getModelStatus } from '../api';

export default function ModelStatus() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelStatus()
      .then(setStatus)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return <div className="status-bar error">API unreachable: {error}</div>;
  }
  if (!status) {
    return <div className="status-bar loading">Checking model...</div>;
  }

  return (
    <div className={`status-bar ${status.ready ? 'ready' : 'not-ready'}`}>
      <span className="dot" />
      {status.ready ? (
        <>
          Model ready &mdash; {status.class_names?.length ?? '?'} classes,
          window {status.window_size ?? '?'} timesteps
        </>
      ) : (
        <>Model not ready: {status.error}</>
      )}
    </div>
  );
}
