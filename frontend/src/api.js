const BASE = import.meta.env.VITE_API_URL || '';

export async function getModelStatus() {
  const res = await fetch(`${BASE}/model/status`);
  if (!res.ok) throw new Error(`Model status fetch failed: ${res.status}`);
  return res.json();
}

export async function submitPredict(window, topK = 3) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ window, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Predict request failed: ${res.status}`);
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job fetch failed: ${res.status}`);
  return res.json();
}

export function parseCSV(text) {
  const lines = text.trim().split('\n').filter(Boolean);
  const rows = lines.map((line) =>
    line.split(',').map((v) => {
      const n = parseFloat(v.trim());
      if (isNaN(n)) throw new Error(`Non-numeric value in CSV: "${v.trim()}"`);
      return n;
    })
  );
  const cols = rows[0].length;
  if (cols !== 12 && cols !== 14) {
    throw new Error(`Expected 12 or 14 columns, got ${cols}`);
  }
  if (rows.length < 10) {
    throw new Error(`Need at least 10 timesteps, got ${rows.length}`);
  }
  return rows;
}

export function generateDemoWindow(timesteps = 200) {
  return Array.from({ length: timesteps }, () =>
    Array.from({ length: 14 }, (_, i) => {
      const isAccel = i < 6;
      const range = isAccel ? 20 : 300;
      return parseFloat(((Math.random() - 0.5) * 2 * range).toFixed(4));
    })
  );
}
