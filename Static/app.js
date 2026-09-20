/**
 * PlacementPredict Client Application (Visual Graph-First Dashboard)
 * Renders all 20 interactive concept graphs across M1, M2, and M3.
 */

let PRESETS = {};
let GRAPHS_LOADED = false;
let GRAPH_DATA = null;
let CHART_INSTANCES = {};
let CURRENT_DATASET_PAGE = 1;
let TOTAL_DATASET_PAGES = 1;
let DATASET_LOADED = false;

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initial fetch of presets
    try {
        const res = await fetch('/api/presets');
        if (res.ok) {
            PRESETS = await res.json();
        }
    } catch (e) {
        console.warn("Could not fetch presets:", e);
    }

    // 2. Load and render all 20 visual graphs immediately
    loadConceptGraphs();

    // 3. Telemetry & initial prediction setup
    fetchMonitoringStats();
    setInterval(fetchMonitoringStats, 10000);
});

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick')?.includes(tabId));
    if (activeBtn) activeBtn.classList.add('active');

    const activePane = document.getElementById(`tab-${tabId}`);
    if (activePane) activePane.classList.add('active');

    if (tabId === 'monitoring') {
        fetchMonitoringStats();
    } else if (tabId === 'graphs') {
        if (!GRAPHS_LOADED) {
            loadConceptGraphs();
        } else {
            setTimeout(() => {
                Object.values(CHART_INSTANCES).forEach(c => c && c.resize && c.resize());
            }, 60);
        }
    } else if (tabId === 'dataset') {
        if (!DATASET_LOADED) {
            fetchDatasetSample(1);
        }
    } else if (tabId === 'predictor') {
        runPrediction();
    }
}

let datasetSearchTimer = null;
function debounceDatasetSearch() {
    clearTimeout(datasetSearchTimer);
    datasetSearchTimer = setTimeout(() => {
        fetchDatasetSample(1);
    }, 250);
}

async function fetchDatasetSample(page = 1) {
    CURRENT_DATASET_PAGE = page;
    const stream = document.getElementById('filterStream')?.value || 'All';
    const status = document.getElementById('filterStatus')?.value || 'All';
    const tier = document.getElementById('filterTier')?.value || 'All';
    const search = document.getElementById('filterSearch')?.value || '';
    const pageSize = document.getElementById('filterPageSize')?.value || '25';

    const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize,
        stream: stream,
        placement_status: status,
        college_tier: tier,
        search: search
    });

    const tbody = document.getElementById('datasetTableBody');
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="15" style="text-align: center; color: #888; padding: 20px;">Fetching records...</td></tr>`;
    }

    try {
        const res = await fetch(`/api/dataset?${params.toString()}`);
        if (!res.ok) return;
        const data = await res.json();
        DATASET_LOADED = true;

        TOTAL_DATASET_PAGES = data.total_pages;
        renderDatasetTable(data);
    } catch (err) {
        console.error("Error fetching dataset sample:", err);
        if (tbody) tbody.innerHTML = `<tr><td colspan="15" style="text-align: center; color: #ef4444;">Failed to load records.</td></tr>`;
    }
}

function renderDatasetTable(data) {
    const tbody = document.getElementById('datasetTableBody');
    const pageInfo = document.getElementById('datasetPageInfo');
    const prevBtn = document.getElementById('btnPrevPage');
    const nextBtn = document.getElementById('btnNextPage');

    if (pageInfo) {
        pageInfo.innerText = `Page ${data.page} of ${data.total_pages} (${data.total_records.toLocaleString()} matching records)`;
    }

    if (prevBtn) prevBtn.disabled = (data.page <= 1);
    if (nextBtn) nextBtn.disabled = (data.page >= data.total_pages);

    if (!tbody) return;
    tbody.innerHTML = '';

    if (!data.records || data.records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="15" style="text-align: center; color: #94a3b8; padding: 20px;">No matching records found.</td></tr>`;
        return;
    }

    data.records.forEach(r => {
        const tr = document.createElement('tr');
        const isPlaced = r.PlacementStatus === 1;
        tr.innerHTML = `
            <td><code>#${r.StudentID}</code></td>
            <td>${r.Gender}</td>
            <td>${r.City}</td>
            <td><span class="badge-tag">${r.CollegeTier}</span></td>
            <td><strong>${r.Stream}</strong></td>
            <td>${r.Specialisation}</td>
            <td><strong>${r.CGPA ?? '-'}</strong></td>
            <td>${r.AttendancePercent ? r.AttendancePercent + '%' : '-'}</td>
            <td>${r.Internships}</td>
            <td>${r.Projects}</td>
            <td>${r.CodingTestScore ?? '<span style="color:#64748b;">N/A</span>'}</td>
            <td>${r.AptitudeTestScore ?? '<span style="color:#64748b;">N/A</span>'}</td>
            <td><span class="${r.HistoryOfBacklogs === 'Yes' ? 'text-red' : 'text-green'}">${r.HistoryOfBacklogs}</span></td>
            <td><span class="${isPlaced ? 'badge-pos' : 'badge-neg'}">${isPlaced ? 'Placed' : 'Not Placed'}</span></td>
            <td><strong>${r.SalaryPackage > 0 ? '₹ ' + r.SalaryPackage + ' LPA' : '₹ 0.0'}</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function changeDatasetPage(delta) {
    const targetPage = CURRENT_DATASET_PAGE + delta;
    if (targetPage >= 1 && targetPage <= TOTAL_DATASET_PAGES) {
        fetchDatasetSample(targetPage);
    }
}

function filterGraphCards(category) {
    document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    if (event && event.target) event.target.classList.add('active');

    document.querySelectorAll('.graph-card').forEach(card => {
        if (category === 'all' || card.getAttribute('data-cat') === category) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });

    setTimeout(() => {
        Object.values(CHART_INSTANCES).forEach(c => c && c.resize && c.resize());
    }, 50);
}

function syncRange(id) {
    const val = document.getElementById(id).value;
    const badge = document.getElementById(`val-${id}`);
    if (badge) {
        badge.innerText = id.includes('Percent') ? `${val}%` : val;
    }
    runPrediction();
}

function loadPreset(presetKey) {
    if (!PRESETS || !PRESETS[presetKey]) return;
    const p = PRESETS[presetKey].data;

    for (const [key, value] of Object.entries(p)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = value;
            const badge = document.getElementById(`val-${key}`);
            if (badge) {
                badge.innerText = key.includes('Percent') ? `${value}%` : value;
            }
        }
    }

    runPrediction();
}

function getFormData(modelOverride = null) {
    const model = modelOverride || document.getElementById('modelSelect')?.value || 'tree';
    const cgpaEl = document.getElementById('CGPA');
    const cgpa = cgpaEl ? parseFloat(cgpaEl.value) : 7.9;
    
    let cgpaTier = "Mid";
    if (cgpa >= 8.0) cgpaTier = "High";
    else if (cgpa < 6.5) cgpaTier = "Low";

    return {
        Gender: "Male",
        City: document.getElementById('City')?.value || "Bangalore",
        CollegeTier: document.getElementById('CollegeTier')?.value || "Tier2",
        Stream: document.getElementById('Stream')?.value || "CS",
        Specialisation: document.getElementById('Specialisation')?.value || "AI",
        Hostel: document.getElementById('Hostel')?.value || "Yes",
        HistoryOfBacklogs: document.getElementById('HistoryOfBacklogs')?.value || "No",
        SGPA_Sem1: cgpa,
        SGPA_Sem2: cgpa,
        SGPA_Sem3: cgpa,
        SGPA_Sem4: cgpa,
        SGPA_Sem5: cgpa,
        SGPA_Sem6: cgpa,
        SGPA_Sem7: cgpa,
        SGPA_Sem8: cgpa,
        CGPA: cgpa,
        AttendancePercent: parseFloat(document.getElementById('AttendancePercent')?.value || 82),
        Internships: parseInt(document.getElementById('Internships')?.value || 1),
        Projects: parseInt(document.getElementById('Projects')?.value || 2),
        Workshops: 1.0,
        Certifications: parseInt(document.getElementById('Certifications')?.value || 2),
        Publications: parseInt(document.getElementById('Publications')?.value || 0),
        AptitudeTestScore: parseFloat(document.getElementById('AptitudeTestScore')?.value || 72),
        SoftSkillsRating: parseFloat(document.getElementById('SoftSkillsRating')?.value || 3.5),
        CodingTestScore: parseFloat(document.getElementById('CodingTestScore')?.value || 68),
        MockInterviewScore: parseFloat(document.getElementById('MockInterviewScore')?.value || 65),
        ExtraCurricular: 1,
        CGPA_Tier: cgpaTier,
        ModelType: model
    };
}

let debounceTimer = null;
function runPrediction() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
        const payload = getFormData();
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.status === 'success') {
                updatePredictionUI(data);
                runCrossModelComparison(payload);
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
    }, 150);
}

function updatePredictionUI(res) {
    const pred = res.prediction;
    const isPlaced = pred.is_placed;
    const probPct = pred.placement_probability_pct;

    const verdictBox = document.getElementById('verdictBox');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictSubtitle = document.getElementById('verdictSubtitle');
    const verdictIcon = document.getElementById('verdictIcon');
    const probValue = document.getElementById('probValue');
    const salaryValue = document.getElementById('salaryValue');
    const riskValue = document.getElementById('riskValue');
    const probBar = document.getElementById('probBar');
    const latencyBadge = document.getElementById('latencyBadge');

    if (latencyBadge) latencyBadge.innerText = `Latency: ${res.inference_latency_ms} ms (${res.model_used})`;
    if (probValue) probValue.innerText = `${probPct}%`;
    if (probBar) probBar.style.width = `${Math.min(Math.max(probPct, 2), 100)}%`;

    if (verdictBox) {
        if (isPlaced) {
            verdictBox.className = 'verdict-box placed';
            if (verdictIcon) verdictIcon.innerText = '🎉';
            if (verdictTitle) verdictTitle.innerText = 'Likely to be Placed';
            if (verdictSubtitle) verdictSubtitle.innerText = 'Profile aligns with top recruiter benchmarks.';
            if (salaryValue) salaryValue.innerText = `₹ ${pred.expected_salary_lpa} LPA`;
            if (riskValue) { riskValue.innerText = pred.risk_level; riskValue.className = 'ribbon-value text-green'; }
            if (probBar) probBar.style.background = 'linear-gradient(90deg, #3b82f6, #10b981)';
        } else {
            verdictBox.className = 'verdict-box not-placed';
            if (verdictIcon) verdictIcon.innerText = '⚠️';
            if (verdictTitle) verdictTitle.innerText = 'Placement At Risk';
            if (verdictSubtitle) verdictSubtitle.innerText = `Placement risk detected. Est. package if placed: ₹ ${pred.potential_salary_if_placed_lpa} LPA`;
            if (salaryValue) salaryValue.innerText = `₹ 0.0 LPA`;
            if (riskValue) { riskValue.innerText = pred.risk_level; riskValue.className = 'ribbon-value text-red'; }
            if (probBar) probBar.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)';
        }
    }

    const factorsList = document.getElementById('factorsList');
    if (factorsList) {
        factorsList.innerHTML = '';
        (res.key_factors || []).forEach(f => {
            const item = document.createElement('div');
            item.className = 'factor-card';
            const isPos = f.score >= 0;
            item.innerHTML = `
                <div><span class="factor-name">${f.feature}</span>: <span style="color: #94a3b8;">${f.value}</span></div>
                <div class="factor-impact ${isPos ? 'pos' : 'neg'}">${f.impact} (${isPos ? '+' : ''}${f.score})</div>
            `;
            factorsList.appendChild(item);
        });
    }
}

async function runCrossModelComparison(basePayload) {
    const grid = document.getElementById('modelCompareGrid');
    const badge = document.getElementById('consensusBadge');
    const currentModel = document.getElementById('modelSelect')?.value || 'lightgbm';

    try {
        const res = await fetch('/api/predict_all', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(basePayload)
        });
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.status === 'success') {
            const consensus = data.consensus;
            if (badge) {
                const isPos = consensus.is_placed;
                badge.innerText = `Consensus: ${consensus.verdict} (${consensus.votes_placed}/${consensus.votes_total} models agree • Avg: ${consensus.average_probability_pct}%)`;
                badge.style.background = isPos ? '#dcfce7' : '#fee2e2';
                badge.style.color = isPos ? '#15803d' : '#b91c1c';
            }

            if (grid && data.models) {
                grid.innerHTML = '';
                data.models.forEach(m => {
                    const card = document.createElement('div');
                    const isActive = (m.key === currentModel) || (currentModel === 'tree' && m.key === 'lightgbm');
                    card.className = `compare-card ${isActive ? 'active-engine' : ''}`;
                    card.style.cursor = 'pointer';
                    card.title = `Click to switch active prediction engine to ${m.name}`;
                    card.onclick = () => {
                        const sel = document.getElementById('modelSelect');
                        if (sel) {
                            sel.value = m.key;
                            runPrediction();
                        }
                    };

                    const isPlaced = m.is_placed;
                    card.innerHTML = `
                        <div class="cmp-header">
                            <div>
                                <div class="cmp-title">${m.name.split(' (')[0]}</div>
                                <div class="cmp-category">${m.category}</div>
                            </div>
                            ${isActive ? '<span style="font-size:10px; background:#2563eb; color:#fff; padding:1px 5px; border-radius:3px; font-weight:700;">ACTIVE</span>' : ''}
                        </div>
                        <div class="cmp-stat">${m.probability_pct}%</div>
                        <div class="cmp-footer">
                            <span class="cmp-tag ${isPlaced ? 'text-green' : 'text-red'}">${m.verdict}</span>
                            <span class="cmp-latency">${m.latency_ms} ms</span>
                        </div>
                    `;
                    grid.appendChild(card);
                });
            }
        }
    } catch (e) {
        console.warn("Could not run multi-model comparison:", e);
    }
}

async function fetchMonitoringStats() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();

        const reqsEl = document.getElementById('monTotalReqs');
        const avgLatEl = document.getElementById('monAvgLatency');
        const p95LatEl = document.getElementById('monP95Latency');
        const placeRateEl = document.getElementById('monPlacementRate');
        const tableBody = document.getElementById('recentLogsBody');

        if (reqsEl) reqsEl.innerText = data.total_requests;
        if (avgLatEl) avgLatEl.innerText = `${data.latency_ms?.avg || 0} ms`;
        if (p95LatEl) p95LatEl.innerText = `${data.latency_ms?.p95 || 0} ms`;
        if (placeRateEl) placeRateEl.innerText = `${data.live_placement_rate_pct}%`;

        if (tableBody && data.recent_requests && data.recent_requests.length > 0) {
            tableBody.innerHTML = '';
            data.recent_requests.slice(0, 10).forEach(log => {
                const row = document.createElement('tr');
                const isPlaced = log.prediction === 'Placed';
                row.innerHTML = `
                    <td><code>${log.timestamp}</code></td>
                    <td>${log.model}</td>
                    <td><span class="${isPlaced ? 'text-green' : 'text-red'}"><strong>${log.prediction}</strong></span></td>
                    <td>${(log.probability * 100).toFixed(1)}%</td>
                    <td>₹ ${log.salary_lpa} LPA</td>
                    <td><code>${log.latency_ms} ms</code></td>
                    <td>${log.cgpa}</td>
                `;
                tableBody.appendChild(row);
            });
        }
    } catch (err) {}
}

// Chart.js Default Light Theme Setup
function getChartDefaultOptions(xTitle = '', yTitle = '') {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#334155', font: { size: 11, weight: '600' } }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.92)',
                titleColor: '#ffffff',
                bodyColor: '#e2e8f0',
                borderColor: '#cbd5e1',
                borderWidth: 1,
                padding: 10,
                cornerRadius: 6
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(0, 0, 0, 0.05)' },
                ticks: { color: '#64748b', font: { size: 10 } },
                title: { display: !!xTitle, text: xTitle, color: '#334155', font: { size: 11, weight: '600' } }
            },
            y: {
                grid: { color: 'rgba(0, 0, 0, 0.05)' },
                ticks: { color: '#64748b', font: { size: 10 } },
                title: { display: !!yTitle, text: yTitle, color: '#334155', font: { size: 11, weight: '600' } }
            }
        }
    };
}

function safeCreateChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (CHART_INSTANCES[canvasId]) {
        try { CHART_INSTANCES[canvasId].destroy(); } catch (e) {}
    }

    try {
        const ctx = canvas.getContext('2d');
        CHART_INSTANCES[canvasId] = new Chart(ctx, config);
    } catch (err) {
        console.error(`Error rendering chart ${canvasId}:`, err);
    }
}

async function loadConceptGraphs() {
    try {
        const res = await fetch('/api/graphs');
        if (!res.ok) return;
        GRAPH_DATA = await res.json();
        GRAPHS_LOADED = true;

        if (typeof Chart === 'undefined') {
            console.warn("Chart.js not yet loaded, retrying in 200ms...");
            setTimeout(loadConceptGraphs, 200);
            return;
        }

        renderAll20ConceptCharts(GRAPH_DATA);
    } catch (e) {
        console.error("Failed to load concept graphs:", e);
    }
}

function renderAll20ConceptCharts(data) {
    if (!data) return;

    // =========================================================================
    // SECTION M1: ML SYSTEM LIFECYCLE GRAPHS
    // =========================================================================

    // 1. CGPA Distribution & Placement Rate
    if (data.cgpa_dist) {
        const d = data.cgpa_dist;
        safeCreateChart('chartM1Cgpa', {
            type: 'bar',
            data: {
                labels: d.labels,
                datasets: [
                    {
                        type: 'line',
                        label: 'Placement Rate (%)',
                        data: d.rate,
                        borderColor: '#38bdf8',
                        borderWidth: 3,
                        yAxisID: 'y1',
                        pointRadius: 4,
                        pointBackgroundColor: '#38bdf8'
                    },
                    {
                        type: 'bar',
                        label: 'Placed Students',
                        data: d.placed,
                        backgroundColor: '#10b981',
                        yAxisID: 'y'
                    },
                    {
                        type: 'bar',
                        label: 'Unplaced Students',
                        data: d.unplaced,
                        backgroundColor: '#ef4444',
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                ...getChartDefaultOptions('CGPA Range', 'Student Count'),
                scales: {
                    x: { grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' } },
                    y: { grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Students', color: '#334155', font: { weight: '600' } } },
                    y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#0284c7' }, title: { display: true, text: 'Placement %', color: '#0284c7', font: { weight: '600' } } }
                }
            }
        });
    }

    // 2. Salary Package by Stream
    if (data.stream_salary) {
        const s = data.stream_salary;
        safeCreateChart('chartM1Stream', {
            type: 'bar',
            data: {
                labels: s.streams,
                datasets: [
                    { label: 'Mean Salary (LPA)', data: s.mean, backgroundColor: '#3b82f6' },
                    { label: 'Max Salary (LPA)', data: s.max, backgroundColor: '#6366f1' }
                ]
            },
            options: getChartDefaultOptions('Engineering Branch', 'Package (₹ LPA)')
        });
    }

    // 3. Batch Latency & Throughput
    if (data.batch_latency) {
        const b = data.batch_latency;
        safeCreateChart('chartM1Batch', {
            type: 'line',
            data: {
                labels: b.batch_sizes.map(x => `${x} items`),
                datasets: [
                    {
                        label: 'Throughput (Reqs / Sec)',
                        data: b.throughput,
                        borderColor: '#10b981',
                        borderWidth: 2.5,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Total Latency (ms)',
                        data: b.latency_ms,
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                ...getChartDefaultOptions('Batch Size', 'Latency (ms)'),
                scales: {
                    x: { grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' } },
                    y: { grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Latency (ms)', color: '#334155', font: { weight: '600' } } },
                    y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#059669' }, title: { display: true, text: 'Throughput (Req/s)', color: '#059669', font: { weight: '600' } } }
                }
            }
        });
    }

    // 4. Training-Serving Drift
    if (data.drift_simulation) {
        const d = data.drift_simulation;
        safeCreateChart('chartM1Drift', {
            type: 'bar',
            data: {
                labels: d.bins,
                datasets: [
                    { label: 'Training Distribution (%)', data: d.training_dist, backgroundColor: '#3b82f6' },
                    { label: 'Live Serving Distribution (%)', data: d.serving_dist, backgroundColor: '#10b981' }
                ]
            },
            options: getChartDefaultOptions('CGPA Feature Bin', 'Share of Population (%)')
        });
    }

    // =========================================================================
    // SECTION M2: LINEAR MODELS AT DEPTH GRAPHS
    // =========================================================================

    // 5. Convergence (Normal Equations vs Gradient Descent)
    if (data.convergence) {
        const c = data.convergence;
        safeCreateChart('chartM2Convergence', {
            type: 'line',
            data: {
                labels: c.epochs,
                datasets: [
                    { label: 'Gradient Descent Loss (Epochs)', data: c.gradient_descent_loss, borderColor: '#60a5fa', borderWidth: 2, pointRadius: 0 },
                    { label: 'Normal Equations Exact Minimum', data: c.normal_equations_optimal, borderColor: '#10b981', borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Training Epochs (t)', 'Loss L(w)')
        });
    }

    // 6. Regularization Geometry
    if (data.regularization_geometry) {
        const reg = data.regularization_geometry;
        safeCreateChart('chartM2Regularization', {
            type: 'line',
            data: {
                datasets: [
                    { label: 'L1 Lasso (Diamond |w1|+|w2|<=1)', data: reg.l1_diamond.x.map((x, i) => ({ x: x, y: reg.l1_diamond.y[i] })), borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)', fill: true, borderWidth: 2, pointRadius: 3 },
                    { label: 'L2 Ridge (Circle w1²+w2²<=1)', data: reg.l2_circle.x.map((x, i) => ({ x: x, y: reg.l2_circle.y[i] })), borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true, borderWidth: 2, pointRadius: 0 },
                    { label: 'Loss Contour Ellipse', data: reg.loss_contours.x.map((x, i) => ({ x: x, y: reg.loss_contours.y[i] })), borderColor: '#a855f7', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Weight w1', 'Weight w2')
        });
    }

    // 7. Sigmoid Function
    if (data.sigmoid) {
        const s = data.sigmoid;
        safeCreateChart('chartM2Sigmoid', {
            type: 'line',
            data: {
                labels: s.z,
                datasets: [{ label: 'Sigmoid Probability P(Y=1|z)', data: s.prob, borderColor: '#60a5fa', backgroundColor: 'rgba(96, 165, 250, 0.1)', fill: true, borderWidth: 2.5, pointRadius: 0 }]
            },
            options: getChartDefaultOptions('Linear Log-Odds z = w·x + b', 'Probability P(Y=1)')
        });
    }

    // 8. Binary Cross-Entropy
    if (data.bce) {
        const b = data.bce;
        safeCreateChart('chartM2Bce', {
            type: 'line',
            data: {
                labels: b.p,
                datasets: [
                    { label: 'Loss when y=1: -log(p)', data: b.loss_y1, borderColor: '#10b981', borderWidth: 2, pointRadius: 0 },
                    { label: 'Loss when y=0: -log(1-p)', data: b.loss_y0, borderColor: '#ef4444', borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Predicted Probability p', 'Log-Loss')
        });
    }

    // 9. Feature Scaling Contours
    if (data.scaling_contours) {
        const sc = data.scaling_contours;
        safeCreateChart('chartM2Scaling', {
            type: 'line',
            data: {
                datasets: [
                    { label: 'Unscaled Contours (Slow Zig-Zag Descent)', data: sc.unscaled_x.map((x, i) => ({ x: x, y: sc.unscaled_y[i] })), borderColor: '#ef4444', borderWidth: 2, pointRadius: 0 },
                    { label: 'StandardScaler Contours (Direct Gradient Descent)', data: sc.scaled_x.map((x, i) => ({ x: x, y: sc.scaled_y[i] })), borderColor: '#10b981', borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Feature Dimension 1', 'Feature Dimension 2')
        });
    }

    // 10. Linear Weights & Odds Ratios
    if (data.linear_drivers) {
        const ld = data.linear_drivers;
        safeCreateChart('chartM2Drivers', {
            type: 'bar',
            data: {
                labels: ld.features,
                datasets: [{ label: 'Regression Weight (w)', data: ld.weights, backgroundColor: ld.colors }]
            },
            options: {
                ...getChartDefaultOptions('Weight Coefficient (w)', 'Feature'),
                indexAxis: 'y'
            }
        });
    }

    // 11. Salary Actual vs Predicted
    if (data.salary_regression) {
        const sp = data.salary_regression;
        safeCreateChart('chartM2SalaryReg', {
            type: 'scatter',
            data: {
                datasets: [
                    { label: 'Candidate Placed Records', data: sp.points, backgroundColor: 'rgba(96, 165, 250, 0.7)', pointRadius: 4 },
                    { type: 'line', label: 'Ideal Fit (y = x)', data: sp.line, borderColor: '#ef4444', borderDash: [4, 4], pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Actual Salary Package (₹ LPA)', 'Predicted Salary Package (₹ LPA)')
        });
    }

    // =========================================================================
    // SECTION M3: TREE-BASED MODELS GRAPHS
    // =========================================================================

    // 12. Tree Splitting Criteria
    if (data.tree_split) {
        const sp = data.tree_split;
        safeCreateChart('chartM3Splitting', {
            type: 'line',
            data: {
                labels: sp.p,
                datasets: [
                    { label: 'Gini Impurity 2p(1-p)', data: sp.gini_impurity, borderColor: '#10b981', borderWidth: 2, pointRadius: 0 },
                    { label: 'Scaled Entropy -1/2 ∑ p log2(p)', data: sp.entropy_scaled, borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0 },
                    { label: 'Misclassification Error min(p, 1-p)', data: sp.misclass_error, borderColor: '#f87171', borderDash: [3, 3], borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Node Class Probability p', 'Impurity Index')
        });
    }

    // 13. CCP Pruning Path
    if (data.ccp_pruning) {
        const ccp = data.ccp_pruning;
        safeCreateChart('chartM3Ccp', {
            type: 'line',
            data: {
                labels: ccp.alpha,
                datasets: [
                    { label: 'Training Accuracy (Bias)', data: ccp.train_accuracy, borderColor: '#60a5fa', borderWidth: 2, pointRadius: 3 },
                    { label: 'Validation Accuracy (Generalization)', data: ccp.val_accuracy, borderColor: '#34d399', borderWidth: 2.5, pointRadius: 3 }
                ]
            },
            options: getChartDefaultOptions('Cost Complexity Parameter α', 'Accuracy (%)')
        });
    }

    // 14. Random Forest OOB Error
    if (data.rf_oob) {
        const rf = data.rf_oob;
        safeCreateChart('chartM3RfOob', {
            type: 'line',
            data: {
                labels: rf.n_trees,
                datasets: [{ label: 'OOB Error Rate (%)', data: rf.oob_error_pct, borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true, borderWidth: 2, pointRadius: 3 }]
            },
            options: getChartDefaultOptions('Number of Estimators (Trees B)', 'OOB Error Rate (%)')
        });
    }

    // 15. Gradient Boosting Loss
    if (data.boosting_loss) {
        const b = data.boosting_loss;
        safeCreateChart('chartM3Boosting', {
            type: 'line',
            data: {
                labels: b.iterations,
                datasets: [
                    { label: 'Train Loss', data: b.train_loss, borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0 },
                    { label: 'Validation Loss', data: b.val_loss, borderColor: '#ec4899', borderWidth: 2, pointRadius: 0 }
                ]
            },
            options: getChartDefaultOptions('Boosting Rounds', 'Log-Loss')
        });
    }

    // 16. ROC Curves Comparison
    if (data.roc_curves) {
        const datasets = [];
        const colors = { 'Ridge Logistic': '#3b82f6', 'LightGBM': '#10b981', 'Random Forest': '#f59e0b' };
        for (const [name, curve] of Object.entries(data.roc_curves)) {
            datasets.push({
                label: `${name} (AUC = ${curve.auc})`,
                data: curve.fpr.map((x, i) => ({ x: x, y: curve.tpr[i] })),
                borderColor: colors[name] || '#94a3b8',
                borderWidth: 2,
                pointRadius: 0
            });
        }
        datasets.push({ label: 'Random Guess', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: 'rgba(0, 0, 0, 0.25)', borderDash: [5, 5], pointRadius: 0 });
        safeCreateChart('chartM3Roc', {
            type: 'line',
            data: { datasets },
            options: getChartDefaultOptions('False Positive Rate (FPR)', 'True Positive Rate (TPR)')
        });
    }

    // 17. Precision-Recall Curves
    if (data.pr_curves) {
        const datasets = [];
        const colors = { 'Ridge Logistic': '#3b82f6', 'LightGBM': '#10b981', 'Random Forest': '#f59e0b' };
        for (const [name, curve] of Object.entries(data.pr_curves)) {
            datasets.push({
                label: `${name} (PR-AUC = ${curve.pr_auc})`,
                data: curve.recall.map((x, i) => ({ x: x, y: curve.precision[i] })),
                borderColor: colors[name] || '#64748b',
                borderWidth: 2,
                pointRadius: 0
            });
        }
        safeCreateChart('chartM3Pr', {
            type: 'line',
            data: { datasets },
            options: getChartDefaultOptions('Recall', 'Precision')
        });
    }

    // 18. Gini MDI vs Permutation Importance
    if (data.feature_importance_comparison) {
        const fi = data.feature_importance_comparison;
        safeCreateChart('chartM3FeatImp', {
            type: 'bar',
            data: {
                labels: fi.features,
                datasets: [
                    { label: 'Gini MDI Importance (%)', data: fi.mdi_importance, backgroundColor: '#10b981' },
                    { label: 'Permutation Importance (%)', data: fi.permutation_importance, backgroundColor: '#3b82f6' }
                ]
            },
            options: {
                ...getChartDefaultOptions('Importance Score (%)', 'Feature Dimension'),
                indexAxis: 'y'
            }
        });
    }

    // 19. Model Comparison Bars
    if (data.model_benchmarks) {
        const mb = data.model_benchmarks;
        safeCreateChart('chartM3ModelBars', {
            type: 'bar',
            data: {
                labels: mb.models,
                datasets: [
                    { label: 'Accuracy (%)', data: mb.accuracy, backgroundColor: '#3b82f6' },
                    { label: 'F1-Score (%)', data: mb.f1_score, backgroundColor: '#10b981' },
                    { label: 'ROC-AUC (%)', data: mb.roc_auc, backgroundColor: '#f59e0b' }
                ]
            },
            options: {
                ...getChartDefaultOptions('Model Architecture', 'Metric Score (%)'),
                scales: {
                    x: { grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' } },
                    y: { min: 85, max: 100, grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { color: '#64748b' }, title: { display: true, text: 'Metric Score (%)', color: '#334155', font: { weight: '600' } } }
                }
            }
        });
    }

    // 20. Tabular Trees vs Neural Networks Radar
    if (data.tabular_benchmark) {
        const tb = data.tabular_benchmark;
        safeCreateChart('chartM3Tabular', {
            type: 'radar',
            data: {
                labels: tb.dimensions,
                datasets: [
                    {
                        label: 'Tree Ensembles (XGB / LGBM)',
                        data: tb.tree_ensembles,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderWidth: 2
                    },
                    {
                        label: 'Neural Networks (MLP)',
                        data: tb.neural_networks,
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.2)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#334155', font: { size: 11, weight: '600' } } }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(0, 0, 0, 0.08)' },
                        grid: { color: 'rgba(0, 0, 0, 0.08)' },
                        pointLabels: { color: '#334155', font: { size: 10, weight: '600' } },
                        ticks: { backdropColor: 'transparent', color: '#64748b' }
                    }
                }
            }
        });
    }
}

// =========================================================================
// UNIVERSAL DATASET ANALYZER & AUTOMATED EDA STUDIO
// =========================================================================

let CURRENT_EDA_DATA = null;
let SELECTED_UPLOAD_FILE = null;

document.addEventListener('DOMContentLoaded', () => {
    // Setup Drag-and-Drop on Analyzer Dropzone
    const dropZone = document.getElementById('analyzerDropZone');
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                SELECTED_UPLOAD_FILE = dt.files[0];
                const label = document.getElementById('selectedFileName');
                if (label) label.innerText = `Selected: ${SELECTED_UPLOAD_FILE.name} (${(SELECTED_UPLOAD_FILE.size / 1024).toFixed(1)} KB)`;
            }
        });
    }
});

function handleFileSelected(e) {
    if (e.target.files && e.target.files.length > 0) {
        SELECTED_UPLOAD_FILE = e.target.files[0];
        const label = document.getElementById('selectedFileName');
        if (label) label.innerText = `Selected: ${SELECTED_UPLOAD_FILE.name} (${(SELECTED_UPLOAD_FILE.size / 1024).toFixed(1)} KB)`;
    }
}

function toggleRawCsvModal() {
    const box = document.getElementById('rawCsvContainer');
    if (box) {
        box.style.display = box.style.display === 'none' ? 'block' : 'none';
    }
}

async function uploadAndAnalyzeDataset() {
    if (!SELECTED_UPLOAD_FILE) {
        alert("Please select or drag & drop a dataset file first.");
        return;
    }

    const btn = document.getElementById('btnAnalyzeUpload');
    if (btn) {
        btn.innerText = "Analyzing Dataset...";
        btn.disabled = true;
    }

    const targetCol = document.getElementById('analyzerTargetCol')?.value || '';
    const formData = new FormData();
    formData.append('file', SELECTED_UPLOAD_FILE);
    if (targetCol) formData.append('target_column', targetCol);

    try {
        const res = await fetch('/api/analyze_upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            renderDatasetAnalysis(data);
        } else {
            alert(`Analysis Error: ${data.detail || 'Could not analyze dataset'}`);
        }
    } catch (err) {
        console.error("Upload error:", err);
        alert("Error connecting to server for analysis.");
    } finally {
        if (btn) {
            btn.innerText = "🚀 Run Deep Analysis";
            btn.disabled = false;
        }
    }
}

async function loadSampleDatasetAnalysis() {
    const targetCol = document.getElementById('analyzerTargetCol')?.value || 'PlacementStatus';
    try {
        const res = await fetch(`/api/analyze_sample?sample_name=placement_50k&target_column=${encodeURIComponent(targetCol)}`);
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            renderDatasetAnalysis(data);
        }
    } catch (err) {
        console.error("Sample dataset error:", err);
    }
}

async function analyzeRawPastedData() {
    const text = document.getElementById('rawCsvText')?.value;
    if (!text || !text.strip ? !text.trim() : !text) {
        alert("Please paste some CSV data.");
        return;
    }
    const targetCol = document.getElementById('analyzerTargetCol')?.value || null;

    try {
        const res = await fetch('/api/analyze_raw_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ csv_text: text, target_column: targetCol })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            renderDatasetAnalysis(data);
        } else {
            alert(`Error: ${data.detail || 'Failed to parse raw data'}`);
        }
    } catch (err) {
        console.error("Raw data error:", err);
    }
}

function renderDatasetAnalysis(data) {
    CURRENT_EDA_DATA = data;
    const container = document.getElementById('analyzerResultsContainer');
    if (container) container.style.display = 'block';

    const s = data.dataset_summary;
    // 1. KPI Ribbon
    const elRows = document.getElementById('edaTotalRows');
    const elCols = document.getElementById('edaTotalCols');
    const elMiss = document.getElementById('edaMissingPct');
    const elDup = document.getElementById('edaDuplicateRows');
    const elMem = document.getElementById('edaMemoryMb');

    if (elRows) elRows.innerText = s.total_rows.toLocaleString();
    if (elCols) elCols.innerText = s.total_columns;
    if (elMiss) elMiss.innerText = `${s.missing_percentage}% (${s.total_missing_cells.toLocaleString()})`;
    if (elDup) elDup.innerText = `${s.duplicate_rows} (${s.duplicate_percentage}%)`;
    if (elMem) elMem.innerText = `${s.memory_mb} MB`;

    // 2. Data Quality Alerts
    const alertsBox = document.getElementById('analyzerAlertsBox');
    const alertsList = document.getElementById('analyzerAlertsList');
    if (alertsBox && alertsList) {
        if (data.data_quality_alerts && data.data_quality_alerts.length > 0) {
            alertsBox.style.display = 'block';
            alertsList.innerHTML = '';
            data.data_quality_alerts.forEach(a => {
                const item = document.createElement('div');
                item.className = `eda-alert-item ${a.type}`;
                const icon = a.type === 'danger' ? '🚨' : (a.type === 'warning' ? '⚠️' : 'ℹ️');
                item.innerHTML = `<strong>${icon} [${a.column}]</strong>: ${a.message}`;
                alertsList.appendChild(item);
            });
        } else {
            alertsBox.style.display = 'block';
            alertsList.innerHTML = `<div class="eda-alert-item info">✅ Clean dataset! No critical data quality issues or extreme anomalies detected.</div>`;
        }
    }

    // 3. Feature Columns Table
    const badge = document.getElementById('edaColCountBadge');
    if (badge) badge.innerText = `${data.columns.length} Features`;
    renderEdaColumnsTable(data.columns);

    // 4. Missing Values Chart
    if (data.missing_chart && data.missing_chart.columns && data.missing_chart.columns.length > 0) {
        safeCreateChart('chartAnalyzerMissing', {
            type: 'bar',
            data: {
                labels: data.missing_chart.columns,
                datasets: [{
                    label: 'Missing Percentage (%)',
                    data: data.missing_chart.percentages,
                    backgroundColor: '#ef4444'
                }]
            },
            options: {
                ...getChartDefaultOptions('Feature', 'Missing (%)'),
                scales: {
                    x: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b', maxRotation: 45 } },
                    y: { min: 0, max: 100, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#64748b' } }
                }
            }
        });
    } else {
        safeCreateChart('chartAnalyzerMissing', {
            type: 'bar',
            data: {
                labels: ['All Features'],
                datasets: [{ label: 'Zero Missing Values', data: [0], backgroundColor: '#10b981' }]
            },
            options: getChartDefaultOptions('', 'Missing (%)')
        });
    }

    // 5. Target Analysis Card & Chart
    const targetCard = document.getElementById('edaTargetCard');
    if (data.target_analysis && targetCard) {
        targetCard.style.display = 'block';
        const t = data.target_analysis;
        const targetTitle = document.getElementById('edaTargetTitle');
        const targetBadge = document.getElementById('edaTargetTypeBadge');
        const targetContent = document.getElementById('edaTargetContent');

        if (targetTitle) targetTitle.innerText = `🎯 Target: ${t.name}`;
        if (targetBadge) targetBadge.innerText = t.type === 'classification' ? `Classification (${t.unique_classes_count} Classes)` : `Regression (Continuous)`;

        if (targetContent) {
            if (t.type === 'classification' && t.classes) {
                targetContent.innerHTML = `
                    <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                        ${t.classes.map(c => `
                            <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:8px 12px; border-radius:6px;">
                                <strong>Class '${c.label}'</strong>: ${c.count.toLocaleString()} (${c.percentage}%)
                            </div>
                        `).join('')}
                    </div>
                `;
            } else if (t.stats) {
                targetContent.innerHTML = `
                    <div style="display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                        <div class="stat-tile" style="padding:8px 12px;"><span class="tile-label">Mean</span><span class="tile-val" style="font-size:14px;">${t.stats.mean}</span></div>
                        <div class="stat-tile" style="padding:8px 12px;"><span class="tile-label">Median</span><span class="tile-val" style="font-size:14px;">${t.stats.median}</span></div>
                        <div class="stat-tile" style="padding:8px 12px;"><span class="tile-label">Min / Max</span><span class="tile-val" style="font-size:14px;">${t.stats.min} - ${t.stats.max}</span></div>
                    </div>
                `;
            }
        }

        // Target associations chart
        if (t.top_features && t.top_features.length > 0) {
            safeCreateChart('chartAnalyzerTarget', {
                type: 'bar',
                data: {
                    labels: t.top_features.map(f => f.feature),
                    datasets: [{
                        label: 'Target Correlation (Pearson r)',
                        data: t.top_features.map(f => f.score),
                        backgroundColor: t.top_features.map(f => f.score >= 0 ? '#3b82f6' : '#f59e0b')
                    }]
                },
                options: {
                    ...getChartDefaultOptions('Feature', 'Correlation Coefficient (r)'),
                    indexAxis: 'y'
                }
            });
        }
    } else if (targetCard) {
        targetCard.style.display = 'none';
    }

    // 6. Correlation Pairs Table
    const corrCard = document.getElementById('edaCorrCard');
    const corrTbody = document.getElementById('edaCorrPairsBody');
    if (data.correlation && data.correlation.top_pairs && data.correlation.top_pairs.length > 0 && corrCard && corrTbody) {
        corrCard.style.display = 'block';
        corrTbody.innerHTML = '';
        data.correlation.top_pairs.forEach(p => {
            const row = document.createElement('tr');
            const isPos = p.correlation >= 0;
            row.innerHTML = `
                <td><strong>${p.feature_a}</strong> & <strong>${p.feature_b}</strong></td>
                <td><code style="font-weight:700; color:${isPos ? '#2563eb' : '#d97706'};">${p.correlation}</code></td>
                <td><span class="badge-tag">${p.strength}</span></td>
            `;
            corrTbody.appendChild(row);
        });
    } else if (corrCard) {
        corrCard.style.display = 'none';
    }

    // 7. Preview Table
    renderEdaPreviewTable(data.preview_records);

    // 8. Auto-inspect the first numerical or prominent column
    if (data.columns && data.columns.length > 0) {
        const firstNum = data.columns.find(c => c.inferred_type === 'numerical') || data.columns[0];
        inspectColumn(firstNum.name);
    }
}

function renderEdaColumnsTable(cols) {
    const tbody = document.getElementById('edaColumnsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!cols || cols.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: #888; padding: 20px;">No matching features.</td></tr>`;
        return;
    }

    cols.forEach(c => {
        const tr = document.createElement('tr');
        let statsSummary = '-';
        if (c.stats) {
            statsSummary = `Mean: <strong>${c.stats.mean}</strong>, Med: ${c.stats.median} [${c.stats.min} - ${c.stats.max}]`;
            if (c.stats.outliers_count > 0) statsSummary += ` • <span style="color:#ef4444;">${c.stats.outliers_count} outliers</span>`;
        } else if (c.top_categories && c.top_categories.length > 0) {
            const top2 = c.top_categories.slice(0, 2).map(x => `${x.value} (${x.pct}%)`).join(', ');
            statsSummary = `Top: ${top2}`;
        }

        tr.innerHTML = `
            <td><strong>${c.name}</strong></td>
            <td><span class="type-pill ${c.inferred_type}">${c.inferred_type}</span></td>
            <td>${c.missing_pct > 0 ? `<span style="color:#ef4444; font-weight:700;">${c.missing_pct}%</span>` : '<span style="color:#10b981;">0%</span>'}</td>
            <td>${c.unique_count.toLocaleString()}</td>
            <td style="font-size: 11px;">${statsSummary}</td>
            <td><button class="col-inspect-btn" onclick="inspectColumn('${c.name}')">🔍 Inspect</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function inspectColumn(colName) {
    if (!CURRENT_EDA_DATA || !CURRENT_EDA_DATA.columns) return;
    const col = CURRENT_EDA_DATA.columns.find(c => c.name === colName);
    if (!col) return;

    const titleEl = document.getElementById('edaActiveChartTitle');
    const subtitleEl = document.getElementById('edaActiveChartSubtitle');
    if (titleEl) titleEl.innerText = `📈 Feature: ${col.name}`;
    if (subtitleEl) subtitleEl.innerText = `Type: ${col.inferred_type.toUpperCase()} • Unique: ${col.unique_count} • Missing: ${col.missing_pct}%`;

    if (col.inferred_type === 'numerical' && col.histogram) {
        safeCreateChart('chartAnalyzerCol', {
            type: 'bar',
            data: {
                labels: col.histogram.labels,
                datasets: [{
                    label: `Frequency Count (${col.name})`,
                    data: col.histogram.counts,
                    backgroundColor: '#3b82f6'
                }]
            },
            options: getChartDefaultOptions(`${col.name} Range`, 'Count')
        });
    } else if (col.top_categories && col.top_categories.length > 0) {
        safeCreateChart('chartAnalyzerCol', {
            type: 'bar',
            data: {
                labels: col.top_categories.map(x => x.value),
                datasets: [{
                    label: `Category Count (${col.name})`,
                    data: col.top_categories.map(x => x.count),
                    backgroundColor: '#10b981'
                }]
            },
            options: getChartDefaultOptions('Category', 'Count')
        });
    }
}

function filterEdaColumns() {
    if (!CURRENT_EDA_DATA || !CURRENT_EDA_DATA.columns) return;
    const q = (document.getElementById('edaColumnSearch')?.value || '').toLowerCase();
    const typeFilter = document.getElementById('edaTypeFilter')?.value || 'all';

    const filtered = CURRENT_EDA_DATA.columns.filter(c => {
        const matchesName = c.name.toLowerCase().includes(q);
        const matchesType = (typeFilter === 'all') || (c.inferred_type === typeFilter);
        return matchesName && matchesType;
    });

    renderEdaColumnsTable(filtered);
}

function renderEdaPreviewTable(records) {
    const thead = document.getElementById('analyzerPreviewThead');
    const tbody = document.getElementById('analyzerPreviewTbody');
    if (!thead || !tbody || !records || records.length === 0) return;

    const cols = Object.keys(records[0]);
    thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
    tbody.innerHTML = '';

    records.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = cols.map(c => `<td>${r[c] !== null ? r[c] : '<span style="color:#94a3b8;">N/A</span>'}</td>`).join('');
        tbody.appendChild(tr);
    });
}
