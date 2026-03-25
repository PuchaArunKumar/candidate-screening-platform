const API_BASE = "";

function $(id) {
  return document.getElementById(id);
}

function setRunId(runId) {
  $("runId").textContent = runId;
  $("btnUploadCandidates").disabled = !runId;
  $("btnSetJob").disabled = !runId;
  $("btnStart").disabled = !runId;
  $("btnUploadTests").disabled = !runId;
  $("btnSchedule").disabled = !runId;
  $("btnSendEmails").disabled = !runId;
  $("btnLeaderboard").disabled = !runId;
  $("btnTestLinks").disabled = !runId;
}

function setStatus(text) {
  $("statusBox").textContent = text;
}

function renderLeaderboardTable(rows) {
  if (!rows || rows.length === 0) {
    $("tableWrap").innerHTML = "<div class='muted'>No data yet.</div>";
    return;
  }
  const cols = Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  for (const c of cols) html += `<th>${c}</th>`;
  html += "</tr></thead><tbody>";
  for (const r of rows) {
    html += "<tr>";
    for (const c of cols) html += `<td>${r[c] ?? ""}</td>`;
    html += "</tr>";
  }
  html += "</tbody></table>";
  $("tableWrap").innerHTML = html;
}

async function api(path, options = {}) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${msg || res.statusText}`);
  }
  return res.json();
}

let currentRunId = "";

$("btnCreateRun").addEventListener("click", async () => {
  $("statusBox").textContent = "Creating run...";
  try {
    const data = await api("/api/pipeline", { method: "POST" });
    currentRunId = data.run_id;
    setRunId(currentRunId);
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnUploadCandidates").addEventListener("click", async () => {
  const file = $("candidatesFile").files[0];
  if (!file) return alert("Choose candidates file");
  const fd = new FormData();
  fd.append("file", file);
  $("statusBox").textContent = "Uploading candidates CSV...";
  try {
    const res = await fetch(API_BASE + `/api/pipeline/${currentRunId}/candidates/csv`, {
      method: "POST",
      body: fd,
    });
    const data = await res.json();
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnSetJob").addEventListener("click", async () => {
  const jd = $("jobDescription").value || "";
  if (!jd.trim()) return alert("Enter job description");
  $("statusBox").textContent = "Setting job description...";
  try {
    const data = await api(`/api/pipeline/${currentRunId}/job-description`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_description: jd }),
    });
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnStart").addEventListener("click", async () => {
  const wResume = parseFloat($("wResume").value);
  const wGithub = parseFloat($("wGithub").value);
  const wCgpa = parseFloat($("wCgpa").value);
  const threshold = parseFloat($("threshold").value);
  const testLinkBase = $("testLinkBase").value;

  $("statusBox").textContent = "Starting pipeline (background)...";
  try {
    const data = await api(`/api/pipeline/${currentRunId}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        weights: { w_resume: wResume, w_github: wGithub, w_cgpa: wCgpa, threshold },
        test_link_base: testLinkBase,
      }),
    });
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnPoll").addEventListener("click", async () => {
  setStatus("Polling...");
  try {
    const data = await api(`/api/pipeline/${currentRunId}`);
    setStatus(JSON.stringify(data, null, 2));
    // If final data exists, enable leaderboard.
    $("btnLeaderboard").disabled = false;
    $("btnTestLinks").disabled = false;
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnLeaderboard").addEventListener("click", async () => {
  try {
    const data = await api(`/api/pipeline/${currentRunId}/leaderboard`);
    const rows = data.rows || [];
    // Remove large fields; this demo only renders basic numeric columns.
    renderLeaderboardTable(
      rows.map((r) => ({
        external_candidate_id: r.external_candidate_id,
        name: r.name,
        email: r.email,
        cgpa: r.cgpa,
        resume_ai_score: r.resume_ai_score,
        github_technical_score: r.github_technical_score,
        overall_score: r.overall_score,
        test_la: r.test_la,
        test_code: r.test_code,
        test_performance_score: r.test_performance_score,
        final_score: r.final_score,
        is_qualified: r.is_qualified,
      }))
    );
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnTestLinks").addEventListener("click", async () => {
  $("testLinksBox").textContent = "Fetching test links...";
  try {
    const data = await api(`/api/pipeline/${currentRunId}/test-links`);
    $("testLinksBox").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("testLinksBox").textContent = String(e);
  }
});

$("btnUploadTests").addEventListener("click", async () => {
  const file = $("testsFile").files[0];
  if (!file) return alert("Choose test results file");
  const test_threshold = parseFloat($("testThreshold").value);

  const fd = new FormData();
  fd.append("file", file);
  $("statusBox").textContent = "Uploading test results...";
  try {
    const res = await fetch(
      API_BASE + `/api/pipeline/${currentRunId}/test-results/csv?test_threshold=${encodeURIComponent(test_threshold)}`,
      { method: "POST", body: fd }
    );
    const data = await res.json();
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnSchedule").addEventListener("click", async () => {
  const start_datetime = $("startDatetime").value;
  const slot_minutes = parseInt($("slotMinutes").value, 10);
  $("statusBox").textContent = "Scheduling interviews (dry-run)...";
  try {
    const data = await api(`/api/pipeline/${currentRunId}/schedule-interviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_datetime,
        slot_minutes,
        timezone: "UTC",
        calendar_id: "primary",
      }),
    });
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

$("btnSendEmails").addEventListener("click", async () => {
  const email_subject = $("emailSubject").value || "Your Technical Test Link";
  $("statusBox").textContent = "Sending emails...";
  try {
    const url =
      `/api/pipeline/${currentRunId}/send-test-emails?email_subject=` +
      encodeURIComponent(email_subject);
    const data = await api(url, { method: "POST" });
    setStatus(JSON.stringify(data, null, 2));
  } catch (e) {
    setStatus(String(e));
  }
});

