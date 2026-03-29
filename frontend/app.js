/**
 * Apex-Oracle — Dashboard Application
 * WebSocket-connected, auto-updating dashboard
 */

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/live';

// ── State ──────────────────────────────────────────────
let currentTicker = 'RELIANCE.NS';
let ws = null;
let priceChart = null;
let shapChart = null;
let reconnectTimer = null;

// ── Initialize ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initStockSelector();
    initRefreshButton();
    initCharts();
    connectWebSocket();
    loadAllData();

    // Refresh data every 30 seconds as fallback
    setInterval(() => loadAllData(), 30000);
});

// ── Stock Selector ────────────────────────────────────
function initStockSelector() {
    const selector = document.getElementById('stock-selector');
    selector.addEventListener('change', (e) => {
        currentTicker = e.target.value;
        loadAllData();
        // Subscribe via WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(`subscribe:${currentTicker}`);
        }
    });
}

// ── Refresh Button ────────────────────────────────────
function initRefreshButton() {
    document.getElementById('btn-refresh').addEventListener('click', async () => {
        const btn = document.getElementById('btn-refresh');
        btn.disabled = true;
        btn.textContent = '⏳';
        try {
            await fetch(`${API_BASE}/api/run-pipeline`, { method: 'POST' });
        } catch (e) {
            console.error('Pipeline trigger failed:', e);
        }
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = '⟳';
        }, 3000);
    });
}

// ── WebSocket ─────────────────────────────────────────
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            setLiveStatus(true);
            ws.send(`subscribe:${currentTicker}`);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'pipeline_complete') {
                    loadAllData();
                    updateTimestamp();
                } else if (data.prediction) {
                    updatePrediction(data);
                }
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        ws.onclose = () => {
            setLiveStatus(false);
            // Reconnect after 5 seconds
            clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(connectWebSocket, 5000);
        };

        ws.onerror = () => {
            setLiveStatus(false);
        };
    } catch (e) {
        setLiveStatus(false);
        reconnectTimer = setTimeout(connectWebSocket, 5000);
    }
}

function setLiveStatus(online) {
    const el = document.getElementById('live-indicator');
    const text = el.querySelector('.live-text');
    el.className = `live-indicator ${online ? 'online' : 'offline'}`;
    text.textContent = online ? 'LIVE' : 'OFFLINE';
}

// ── Load All Data ────────────────────────────────────
async function loadAllData() {
    await Promise.allSettled([
        loadPrediction(),
        loadHistorical(),
        loadAlerts(),
        loadSentiment(),
        loadExplainability(),
        loadResults(),
        loadStatus(),
    ]);
}

// ── Prediction ────────────────────────────────────────
async function loadPrediction() {
    try {
        const res = await fetch(`${API_BASE}/api/prediction/${currentTicker}`);
        const data = await res.json();
        if (!data.error) updatePrediction(data);
    } catch (e) { /* API not ready yet */ }
}

function updatePrediction(data) {
    const pred = data.prediction || {};
    const regime = data.regime || {};
    const agreement = data.agreement || {};
    const models = data.models || {};

    // Company name
    const companyEl = document.getElementById('pred-company');
    companyEl.textContent = data.company || currentTicker;
    document.getElementById('pred-ticker').textContent = data.ticker || currentTicker;

    // Prices
    const currentPrice = pred.current_price || 0;
    const predictedPrice = pred.price || 0;
    const changePct = pred.change_pct || 0;
    const direction = pred.direction || '—';

    document.getElementById('current-price').textContent = `₹${currentPrice.toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
    document.getElementById('predicted-price').textContent = `₹${predictedPrice.toLocaleString('en-IN', {maximumFractionDigits: 2})}`;

    // Direction arrow
    const arrowEl = document.getElementById('direction-arrow');
    arrowEl.className = `arrow-block ${direction === 'UP' ? 'up' : 'down'}`;
    arrowEl.querySelector('.arrow').textContent = direction === 'UP' ? '↗' : '↘';

    // Change %
    const changeEl = document.getElementById('change-pct');
    changeEl.textContent = `${changePct > 0 ? '+' : ''}${changePct.toFixed(2)}%`;
    changeEl.className = `change ${changePct >= 0 ? 'positive' : 'negative'}`;

    // Direction badge
    const badge = document.getElementById('direction-badge');
    badge.textContent = direction;
    badge.className = `direction-badge ${direction === 'UP' ? 'up' : 'down'}`;

    // Confidence ring
    const confidence = pred.confidence || 0;
    const ring = document.getElementById('confidence-ring');
    const circumference = 2 * Math.PI * 52;
    ring.style.strokeDashoffset = circumference * (1 - confidence);

    if (confidence >= 0.8) ring.style.stroke = '#10b981';
    else if (confidence >= 0.6) ring.style.stroke = '#6366f1';
    else ring.style.stroke = '#f59e0b';

    document.getElementById('confidence-value').textContent = `${(confidence * 100).toFixed(0)}%`;

    // Agreement
    document.getElementById('models-agree').textContent =
        `${agreement.models_agreeing || 0} / ${agreement.total_models || 8}`;
    document.getElementById('classifier-vote').textContent =
        agreement.classification_vote || '—';

    const conflictRow = document.getElementById('conflict-row');
    conflictRow.style.display = agreement.conflict_flag ? 'flex' : 'none';

    // Regime
    updateRegime(regime);

    // Models
    updateModels(models, pred);

    // Timestamp
    updateTimestamp();
}

// ── Regime ─────────────────────────────────────────────
function updateRegime(regime) {
    const name = regime.current || 'UNKNOWN';
    const confidence = regime.confidence || 0;
    const probs = regime.probabilities || {};

    const badge = document.getElementById('regime-badge');
    const regimeClass = name === 'BULL' ? 'bull' : name === 'BEAR' ? 'bear'
        : name === 'SIDEWAYS' ? 'sideways' : 'volatile';
    badge.className = `regime-badge ${regimeClass}`;

    const icons = { BULL: '🐂', BEAR: '🐻', SIDEWAYS: '↔️', HIGH_VOLATILITY: '🌊' };
    document.getElementById('regime-icon').textContent = icons[name] || '◎';
    document.getElementById('regime-name').textContent = name.replace('_', ' ');

    document.getElementById('regime-confidence-bar').style.width = `${confidence * 100}%`;
    document.getElementById('regime-confidence-value').textContent = `${(confidence * 100).toFixed(0)}%`;

    // Probability bars
    const probMap = { bull: 'BULL', bear: 'BEAR', sideways: 'SIDEWAYS', volatile: 'HIGH_VOLATILITY' };
    Object.entries(probMap).forEach(([cssId, key]) => {
        const el = document.getElementById(`prob-${cssId}`);
        if (el) el.style.width = `${(probs[key] || 0) * 100}%`;
    });
}

// ── Models ─────────────────────────────────────────────
function updateModels(models, pred) {
    const regression = models.regression || {};
    const classification = models.classification || {};
    const weights = models.regression_weights || {};

    // Regression
    const regModels = ['arima', 'garch', 'lstm', 'xgboost'];
    regModels.forEach(name => {
        const m = regression[name];
        const predEl = document.getElementById(`model-${name}-pred`);
        const weightEl = document.getElementById(`model-${name}-weight`);

        if (m) {
            predEl.textContent = `₹${(m.prediction || 0).toLocaleString('en-IN', {maximumFractionDigits: 2})}`;
            predEl.style.color = m.direction === 'UP' ? '#10b981' : '#ef4444';
        }
        if (weights[name] !== undefined) {
            weightEl.textContent = `${(weights[name] * 100).toFixed(0)}%`;
        }
    });

    // Classification
    const classMap = {
        naive_bayes: 'nb', svm: 'svm',
        random_forest: 'rf', logistic_regression: 'lr'
    };
    Object.entries(classMap).forEach(([key, abbr]) => {
        const m = classification[key];
        const dirEl = document.getElementById(`model-${abbr}-dir`);
        const probEl = document.getElementById(`model-${abbr}-prob`);

        if (m) {
            dirEl.textContent = m.direction || '—';
            dirEl.className = `model-direction ${(m.direction || '').toLowerCase()}`;
            probEl.textContent = `${((m.probability_up || 0) * 100).toFixed(0)}%`;
        }
    });
}

// ── Historical Price Chart ────────────────────────────
async function loadHistorical() {
    try {
        const res = await fetch(`${API_BASE}/api/historical/${currentTicker}?limit=100`);
        const data = await res.json();
        if (data.data) updatePriceChart(data.data);
    } catch (e) { /* API not ready */ }
}

function initCharts() {
    // Price chart
    const ctx = document.getElementById('price-chart').getContext('2d');
    priceChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                },
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', maxTicksLimit: 10, font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
            },
        },
    });

    // SHAP chart
    const ctx2 = document.getElementById('shap-chart').getContext('2d');
    shapChart = new Chart(ctx2, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    ticks: { color: '#94a3b8', font: { size: 10 } },
                    grid: { display: false },
                },
            },
        },
    });
}

function updatePriceChart(data) {
    const labels = data.map(d => {
        const date = new Date(d.Date);
        return date.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
    });
    const prices = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume);

    priceChart.data.labels = labels;
    priceChart.data.datasets = [
        {
            label: 'Close Price',
            data: prices,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4,
        },
    ];
    priceChart.update('none');
}

// ── Alerts ─────────────────────────────────────────────
async function loadAlerts() {
    try {
        const res = await fetch(`${API_BASE}/api/alerts?limit=15`);
        const data = await res.json();
        if (data.alerts) updateAlerts(data.alerts);
    } catch (e) { /* API not ready */ }
}

function updateAlerts(alerts) {
    const feed = document.getElementById('alerts-feed');
    const countEl = document.getElementById('alert-count');
    countEl.textContent = alerts.length;

    if (alerts.length === 0) {
        feed.innerHTML = '<p class="placeholder">No alerts yet. Run the pipeline to generate predictions.</p>';
        return;
    }

    feed.innerHTML = alerts.reverse().map(a => `
        <div class="alert-item ${a.severity || 'info'}">
            <div>
                <div class="alert-title">${a.title || ''}</div>
                <div class="alert-message">${a.message || ''}</div>
                <div class="alert-time">${formatTime(a.timestamp)}</div>
            </div>
        </div>
    `).join('');
}

// ── Sentiment ──────────────────────────────────────────
async function loadSentiment() {
    try {
        const res = await fetch(`${API_BASE}/api/sentiment/${currentTicker}`);
        const data = await res.json();
        updateSentiment(data);
    } catch (e) { /* API not ready */ }
}

function updateSentiment(data) {
    const latest = data.latest || {};
    const score = latest.sentiment_score || 0;

    // Update gauge (score ranges from -1 to 1 → map to 0-100%)
    const position = ((score + 1) / 2) * 100;
    document.getElementById('sentiment-fill').style.left = `${position}%`;
    document.getElementById('sentiment-value').textContent = score.toFixed(3);

    const label = score > 0.1 ? 'Positive' : score < -0.1 ? 'Negative' : 'Neutral';
    document.getElementById('sentiment-label').textContent = label;
    document.getElementById('sentiment-label').style.left = `${position}%`;

    // Headlines
    const headlines = data.headlines || [];
    const container = document.getElementById('sentiment-headlines');
    if (headlines.length > 0) {
        container.innerHTML = headlines.slice(0, 6).map(h => `
            <div class="headline-item">${h.title || ''}</div>
        `).join('');
    }
}

// ── Explainability (SHAP) ──────────────────────────────
async function loadExplainability() {
    try {
        const res = await fetch(`${API_BASE}/api/explainability/${currentTicker}`);
        const data = await res.json();
        if (!data.error) updateExplainability(data);
    } catch (e) { /* API not ready */ }
}

function updateExplainability(data) {
    // Narrative
    const narrativeEl = document.getElementById('narrative-text');
    if (data.narrative) {
        // Simple markdown-like rendering
        let html = data.narrative
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        narrativeEl.innerHTML = html;
    }

    // SHAP chart
    const shap = data.shap || {};
    const features = shap.top_features || [];
    if (features.length > 0) {
        const labels = features.map(f => f.display_name || f.feature);
        const values = features.map(f => f.impact);
        const colors = values.map(v => v > 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');

        shapChart.data.labels = labels;
        shapChart.data.datasets = [{
            data: values,
            backgroundColor: colors,
            borderRadius: 4,
        }];
        shapChart.update('none');
    }
}

// ── Status ─────────────────────────────────────────────
async function loadStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        const sched = data.scheduler || {};

        document.getElementById('footer-scheduler-status').textContent =
            `Scheduler: ${sched.running ? '✅ Running' : '❌ Stopped'}`;
        document.getElementById('footer-memory').textContent =
            `Memory: ${(sched.memory_percent || 0).toFixed(0)}%`;
        document.getElementById('footer-clients').textContent =
            `Clients: ${data.connected_clients || 0}`;
    } catch (e) { /* API not ready */ }
}

// ── Backtesting Results ────────────────────────────────
async function loadResults() {
    try {
        const res = await fetch(`${API_BASE}/api/results`);
        const data = await res.json();
        if (!data.error) updateResults(data);
    } catch (e) { /* API not ready */ }
}

function updateResults(data) {
    const tbody = document.getElementById('results-tbody');
    if (!tbody || Object.keys(data).length === 0) return;
    
    let html = '';
    const models = ['XGBoost', 'RandomForest', 'LogisticRegression', 'SVM', 'NaiveBayes', 'LSTM'];
    
    for (const [stock, results] of Object.entries(data)) {
        let firstRowForStock = true;
        for (const model of models) {
            const metrics = results[model];
            if (!metrics) continue;
            
            let mae = '-';
            let mape = '-';
            let acc = '-';

            if (metrics.error) {
                mae = `<span style="color: #ef4444; font-size: 0.8rem;">Error</span>`;
            } else {
                if (metrics.MAE !== undefined && metrics.MAE !== null) mae = metrics.MAE.toFixed(2);
                if (metrics.MAPE !== undefined && metrics.MAPE !== null) mape = metrics.MAPE.toFixed(2) + '%';
                if (metrics.Accuracy !== undefined && metrics.Accuracy !== null) acc = metrics.Accuracy.toFixed(2) + '%';
            }
            
            html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.05); transition: background-color 0.2s;" onmouseover="this.style.backgroundColor='rgba(255,255,255,0.03)'" onmouseout="this.style.backgroundColor='transparent'">
                <td style="padding: 0.75rem; font-weight: 500;">${firstRowForStock ? stock : ''}</td>
                <td style="padding: 0.75rem; color: #cbd5e1;">${model}</td>
                <td style="padding: 0.75rem;">${mae}</td>
                <td style="padding: 0.75rem;">${mape}</td>
                <td style="padding: 0.75rem;">${acc}</td>
            </tr>`;
            firstRowForStock = false;
        }
    }
    
    tbody.innerHTML = html;
}

// ── Helpers ────────────────────────────────────────────
function updateTimestamp() {
    const now = new Date();
    document.getElementById('update-time').textContent =
        now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleString('en-IN', {
        day: '2-digit', month: 'short',
        hour: '2-digit', minute: '2-digit',
    });
}
