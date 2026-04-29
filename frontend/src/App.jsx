import React, { useState } from 'react';
import ModelStatus from './components/ModelStatus';
import PredictForm from './components/PredictForm';
import ResultDisplay from './components/ResultDisplay';

export default function App() {
  const [jobId, setJobId] = useState(null);

  return (
    <div className="app">
      <header>
        <h1>PawAI</h1>
        <p className="tagline">Pet Behavior Classification from IMU Sensors</p>
        <ModelStatus />
      </header>

      <main>
        {jobId ? (
          <ResultDisplay jobId={jobId} onReset={() => setJobId(null)} />
        ) : (
          <PredictForm onJobSubmitted={setJobId} />
        )}
      </main>
    </div>
  );
}
