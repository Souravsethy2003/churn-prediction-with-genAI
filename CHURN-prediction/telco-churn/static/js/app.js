// app/static/js/app.js
let latestTopReasons = [];
let latestRowsScored = 0;
let latestChurnRate = null;

const fileInput = document.getElementById('fileInput');
const fileNameEl = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const bulkForm = document.getElementById('bulkForm');
const bulkOutput = document.getElementById('bulkOutput');
const downloadLink = document.getElementById('downloadLink');
const resultArea = document.getElementById('resultArea');
const churnChart = document.getElementById('churnChart');
const topReasons = document.getElementById('topReasons');
const previewTable = document.getElementById('previewTable');
const genaiBtn = document.getElementById('genaiBtn');
const genaiOutput = document.getElementById('genaiOutput');

// show filename and enable button
fileInput.addEventListener('change', (e) => {
  const f = fileInput.files[0];
  if (f) {
    fileNameEl.textContent = f.name;
    uploadBtn.disabled = false;
  } else {
    fileNameEl.textContent = 'No file chosen';
    uploadBtn.disabled = true;
  }
});

// reset UI helper
function resetUI() {
  resultArea.style.display = 'none';
  bulkOutput.textContent = '';
  downloadLink.style.display = 'none';
  churnChart.src = '';
  topReasons.innerHTML = '';
  previewTable.innerHTML = '';
  genaiOutput.textContent = 'Click "Generate AI Recommendations" to get actionable items.';
  latestTopReasons = [];
  latestRowsScored = 0;
  latestChurnRate = null;
}

bulkForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  resetUI();
  const f = fileInput.files[0];
  if (!f) {
    bulkOutput.textContent = 'Please choose a CSV file.';
    return;
  }

  const fd = new FormData();
  fd.append('file', f);

  bulkOutput.textContent = 'Uploading and scoring...';
  uploadBtn.disabled = true;

  try {
    const res = await fetch('/api/bulk_predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      bulkOutput.textContent = 'Server error: ' + txt;
      uploadBtn.disabled = false;
      return;
    }

    const data = await res.json();
    latestTopReasons = data.top_reasons || [];
    latestRowsScored = data.rows_scored || (data.preview ? data.preview.length : 0);
    if (data.preview && data.preview.length) {
      const churnCount = data.preview.filter(r => r.prediction === 1).length;
      latestChurnRate = (churnCount / data.preview.length);
    }

    if (data.download_url) {
      downloadLink.href = data.download_url;
      downloadLink.style.display = 'inline-block';
      downloadLink.textContent = `Download scored CSV (${data.rows_scored || '—'} rows)`;
    }

    if (data.chart_url) {
      churnChart.src = data.chart_url + '?t=' + Date.now();
    }

    // top reasons list
    topReasons.innerHTML = '';
    if (latestTopReasons.length) {
      latestTopReasons.forEach(function(item, idx) {
        const li = document.createElement('li');
        li.textContent = (idx+1) + '. ' + item[0] + ' — ' + item[1];
        topReasons.appendChild(li);
      });
    } else {
      topReasons.innerHTML = '<li>No strong drivers detected</li>';
    }

    // preview table
    previewTable.innerHTML = '';
    if (data.preview && data.preview.length) {
      const cols = Object.keys(data.preview[0]);
      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const thr = document.createElement('tr');
      cols.forEach(c => {
        const th = document.createElement('th');
        th.textContent = c;
        thr.appendChild(th);
      });
      thead.appendChild(thr); table.appendChild(thead);

      const tbody = document.createElement('tbody');
      data.preview.forEach(row => {
        const tr = document.createElement('tr');
        cols.forEach(c => {
          const td = document.createElement('td');
          td.textContent = row[c];
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      previewTable.appendChild(table);
    } else {
      previewTable.textContent = 'No preview available';
    }

    resultArea.style.display = 'block';
    bulkOutput.textContent = 'Done';
  } catch (err) {
    console.error(err);
    bulkOutput.textContent = 'Error: ' + err.message;
  } finally {
    uploadBtn.disabled = false;
  }
});

// Parse LLM text into simple HTML (headlines and bullets). Basic safe parser.
function renderGenAIText(raw) {
  if (!raw) return '<div class="muted">No recommendation returned.</div>';
  // escape HTML
  const esc = (s) => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const lines = String(raw).split('\n');
  let html = '';
  let inList = false;
  for (let ln of lines) {
    ln = ln.trim();
    if (!ln) { if (inList) { html += '</ul>'; inList = false } html += '<br/>'; continue; }
    // headings like "### " or "1) " -> render as bold headline
    if (ln.startsWith('### ') || /^\d+\.\s/.test(ln) || /^Headline[:\-]/i.test(ln)) {
      if (inList) { html += '</ul>'; inList = false }
      html += `<h4>${esc(ln.replace(/^#+\s*/, ''))}</h4>`;
      continue;
    }
    // bullets starting with '-' or '*' or '*   '
    if (/^[-*]\s+/.test(ln) || /^\u2022\s+/.test(ln) || /^\d+\)/.test(ln)) {
      if (!inList) { html += '<ul>'; inList = true; }
      // bold **text**
      let ln2 = esc(ln.replace(/^[-*]\s+/, ''));
      ln2 = ln2.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      html += `<li>${ln2}</li>`;
      continue;
    }
    // lines that contain "Action:" or "Why:" -> emphasize label
    if (/^(Action|Why|Estimated effort|Expected impact|KPI)\s*[:\-]/i.test(ln)) {
      if (!inList) { html += '<ul>'; inList = true; }
      let ln2 = esc(ln).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      html += `<li>${ln2}</li>`;
      continue;
    }
    // fallback: paragraph
    if (inList) { html += '</ul>'; inList = false }
    html += `<p>${esc(ln).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</p>`;
  }
  if (inList) html += '</ul>';
  return html;
}

async function fetchGenaiRecs() {
  if (!latestTopReasons || latestTopReasons.length === 0) {
    genaiOutput.innerHTML = '<div class="muted">Run a bulk prediction first to generate top reasons.</div>';
    return;
  }
  genaiOutput.textContent = '';
  genaiBtn.disabled = true;
  genaiBtn.textContent = 'Generating…';

  const payload = {
    top_reasons: latestTopReasons,
    rows_scored: latestRowsScored,
    churn_rate_est: latestChurnRate
  };

  try {
    const res = await fetch('/api/genai_recs', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const txt = await res.text();
      genaiOutput.textContent = 'GenAI error: ' + txt;
      return;
    }
    const j = await res.json();
    // render nicely
    genaiOutput.innerHTML = renderGenAIText(j.text || '');
  } catch (err) {
    console.error(err);
    genaiOutput.textContent = 'GenAI call failed: ' + err.message;
  } finally {
    genaiBtn.disabled = false;
    genaiBtn.textContent = 'Generate AI Recommendations';
  }
}

genaiBtn.addEventListener('click', fetchGenaiRecs);
