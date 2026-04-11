const state = {
  modules: [
    { id: 0, name: "python_basics" },
    { id: 1, name: "data_structures" },
    { id: 2, name: "ml_fundamentals" },
    { id: 3, name: "deep_learning" },
    { id: 4, name: "nlp_basics" },
    { id: 5, name: "reinforcement_learning" }
  ]
};

const els = {
  baseUrl: document.getElementById("baseUrl"),
  apiKey: document.getElementById("apiKey"),
  heroStatus: document.getElementById("heroStatus"),
  healthBtn: document.getElementById("healthBtn"),
  healthOut: document.getElementById("healthOut"),
  learnerId: document.getElementById("learnerId"),
  explain: document.getElementById("explain"),
  addRowBtn: document.getElementById("addRowBtn"),
  sampleBtn: document.getElementById("sampleBtn"),
  predictBtn: document.getElementById("predictBtn"),
  sessionBody: document.getElementById("sessionBody"),
  predictOut: document.getElementById("predictOut"),
  batchSize: document.getElementById("batchSize"),
  batchSessions: document.getElementById("batchSessions"),
  batchSeed: document.getElementById("batchSeed"),
  genBatchBtn: document.getElementById("genBatchBtn"),
  runBatchBtn: document.getElementById("runBatchBtn"),
  batchInput: document.getElementById("batchInput"),
  batchOut: document.getElementById("batchOut")
};

function baseUrl() {
  return els.baseUrl.value.trim().replace(/\/$/, "");
}

function authHeaders() {
  const h = { "Content-Type": "application/json" };
  const token = els.apiKey.value.trim();
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

async function api(path, options = {}) {
  const res = await fetch(`${baseUrl()}${path}`, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {})
    }
  });

  const text = await res.text();
  let body;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { raw: text };
  }

  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`);
    err.payload = body;
    throw err;
  }
  return body;
}

function print(outEl, data, kind = "ok") {
  outEl.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
  outEl.classList.remove("ok", "err");
  outEl.classList.add(kind);
}

function rowTemplate(seed = {}) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>
      <select data-k="module_id">
        ${state.modules.map((m) => `<option value="${m.id}">${m.id} - ${m.name}</option>`).join("")}
      </select>
    </td>
    <td><input data-k="quiz_score" type="number" min="0" max="100" step="0.1" value="${seed.quiz_score ?? 70}"></td>
    <td><input data-k="engagement_rate" type="number" min="0" max="1" step="0.01" value="${seed.engagement_rate ?? 0.72}"></td>
    <td><input data-k="hint_count" type="number" min="0" max="10" step="1" value="${seed.hint_count ?? 2}"></td>
    <td><input data-k="session_duration" type="number" min="0" max="120" step="0.5" value="${seed.session_duration ?? 42}"></td>
    <td><input data-k="correct_attempts" type="number" min="0" max="50" step="1" value="${seed.correct_attempts ?? 11}"></td>
    <td><input data-k="incorrect_attempts" type="number" min="0" max="50" step="1" value="${seed.incorrect_attempts ?? 3}"></td>
    <td><button class="ghost" type="button" data-act="remove">x</button></td>
  `;

  tr.querySelector("select").value = String(seed.module_id ?? 0);
  tr.querySelector('[data-act="remove"]').addEventListener("click", () => tr.remove());
  return tr;
}

function addRows(n = 1) {
  for (let i = 0; i < n; i++) {
    els.sessionBody.appendChild(rowTemplate());
  }
}

function collectSessions() {
  const rows = [...els.sessionBody.querySelectorAll("tr")];
  if (!rows.length) throw new Error("Add at least one session row");

  return rows.map((tr) => {
    const obj = {};
    tr.querySelectorAll("[data-k]").forEach((input) => {
      const key = input.getAttribute("data-k");
      const raw = input.value;
      obj[key] = key.includes("rate") || key.includes("score") || key.includes("duration")
        ? Number(raw)
        : parseInt(raw, 10);
      if (key === "module_id") obj[key] = parseInt(raw, 10);
    });
    return obj;
  });
}

function random(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function buildBatchPayload(nLearners, nSessions, seed) {
  const learners = [];
  for (let i = 0; i < nLearners; i++) {
    const sessions = [];
    for (let s = 0; s < nSessions; s++) {
      const r = random(seed + i * 13 + s * 7);
      const score = Math.round((45 + r * 50) * 10) / 10;
      const engage = Math.round((0.25 + r * 0.72) * 100) / 100;
      const hint = Math.floor(1 + r * 7);
      sessions.push({
        module_id: (s + i) % 6,
        quiz_score: Math.min(100, score),
        engagement_rate: Math.min(1, engage),
        hint_count: hint,
        session_duration: Math.round((20 + r * 50) * 10) / 10,
        correct_attempts: Math.floor(4 + r * 20),
        incorrect_attempts: Math.floor(1 + (1 - r) * 12)
      });
    }
    learners.push({
      learner_id: `batch-user-${String(i + 1).padStart(2, "0")}`,
      sessions
    });
  }
  return { learners };
}

function summarizeBatch(response) {
  const byTier = { low: 0, medium: 0, high: 0 };
  const byIntervention = {};

  for (const r of response.results || []) {
    byTier[r.dropout_tier] = (byTier[r.dropout_tier] || 0) + 1;
    byIntervention[r.intervention] = (byIntervention[r.intervention] || 0) + 1;
  }

  return {
    total: response.total,
    failed: response.failed,
    risk_tiers: byTier,
    interventions: byIntervention,
    first_result: response.results?.[0] || null
  };
}

els.healthBtn.addEventListener("click", async () => {
  try {
    const data = await api("/health", { method: "GET" });
    print(els.healthOut, data, "ok");
    els.heroStatus.textContent = data.model_loaded ? `API ready (${data.model_version})` : "API unavailable";
  } catch (e) {
    print(els.healthOut, e.payload || e.message, "err");
    els.heroStatus.textContent = "API error";
  }
});

els.addRowBtn.addEventListener("click", () => addRows(1));

els.sampleBtn.addEventListener("click", () => {
  els.sessionBody.innerHTML = "";
  const sample = [
    { module_id: 0, quiz_score: 58, engagement_rate: 0.56, hint_count: 4, session_duration: 38, correct_attempts: 6, incorrect_attempts: 7 },
    { module_id: 1, quiz_score: 63, engagement_rate: 0.62, hint_count: 3, session_duration: 41, correct_attempts: 8, incorrect_attempts: 5 },
    { module_id: 2, quiz_score: 69, engagement_rate: 0.7, hint_count: 2, session_duration: 45, correct_attempts: 10, incorrect_attempts: 4 },
    { module_id: 3, quiz_score: 73, engagement_rate: 0.75, hint_count: 2, session_duration: 47, correct_attempts: 12, incorrect_attempts: 3 },
    { module_id: 4, quiz_score: 79, engagement_rate: 0.82, hint_count: 1, session_duration: 50, correct_attempts: 15, incorrect_attempts: 2 }
  ];
  sample.forEach((s) => els.sessionBody.appendChild(rowTemplate(s)));
});

els.predictBtn.addEventListener("click", async () => {
  try {
    const payload = {
      learner_id: els.learnerId.value.trim() || "demo-learner-001",
      sessions: collectSessions()
    };

    const explainQ = els.explain.checked ? "?explain=true" : "";
    const data = await api(`/predict/learner${explainQ}`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
    print(els.predictOut, data, "ok");
  } catch (e) {
    print(els.predictOut, e.payload || e.message, "err");
  }
});

els.genBatchBtn.addEventListener("click", () => {
  const payload = buildBatchPayload(
    Number(els.batchSize.value || 5),
    Number(els.batchSessions.value || 6),
    Number(els.batchSeed.value || 21)
  );
  els.batchInput.value = JSON.stringify(payload, null, 2);
});

els.runBatchBtn.addEventListener("click", async () => {
  try {
    const raw = els.batchInput.value.trim();
    if (!raw) throw new Error("Generate or paste batch JSON first");

    const payload = JSON.parse(raw);
    const data = await api("/predict/batch", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    print(els.batchOut, summarizeBatch(data), "ok");
  } catch (e) {
    const fallback = e.payload || e.message;
    print(els.batchOut, fallback, "err");
  }
});

addRows(3);
els.genBatchBtn.click();
