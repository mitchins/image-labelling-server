/**
 * Smart Label - Application Logic
 * Zero-friction labeling with keyboard shortcuts and preloading
 */

// Utility: Generate UUID if crypto.randomUUID is not available
function generateUUID() {
    if (crypto && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Fallback UUID v4 generator
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// State
let currentImage = null;
let preloadedImages = [];
let isLabeling = false;
let sessionId = localStorage.getItem('smartLabelSession') || generateUUID();
localStorage.setItem('smartLabelSession', sessionId);

// Config (loaded from server)
let CONFIG = null;
let STYLES = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro'];
let STYLE_COLORS = {};
let KEY_MAP = {};

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        CONFIG = await res.json();
        
        STYLES = CONFIG.labels || STYLES;
        STYLE_COLORS = CONFIG.label_colors || {};
        
        // Build key map: 1-9 for labels, x/space for refuse, q for bad quality
        KEY_MAP = {};
        STYLES.forEach((s, i) => {
            if (i < 9) KEY_MAP[String(i + 1)] = s;
        });
        KEY_MAP['x'] = 'REFUSE';
        KEY_MAP[' '] = 'REFUSE';
        KEY_MAP['q'] = 'BAD_QUALITY';
        
        // Update UI with config
        document.getElementById('taskName').textContent = CONFIG.name || 'Labeling Task';
        updateShortcutsDisplay();
        
    } catch (err) {
        console.warn('Failed to load config, using defaults:', err);
    }
}

function updateShortcutsDisplay() {
    const shortcuts = document.getElementById('shortcuts');
    if (!shortcuts) return;
    
    const groups = STYLES.map((s, i) => 
        `<span class="shortcut-group"><kbd>${i + 1}</kbd> ${capitalize(s)}</span>`
    ).join('');
    
    shortcuts.innerHTML = groups + 
        ' <span class="shortcut-group"><kbd>X</kbd> Refuse</span>' +
        ' <span class="shortcut-group"><kbd>Q</kbd> Bad Quality</span>' +
        ' <span class="shortcut-group"><kbd>Z</kbd> Undo</span>' +
        ' <span class="shortcut-group"><kbd>H</kbd> History</span>';
}

// ============================================================================
// Core Functions
// ============================================================================

async function loadNext() {
    try {
        const res = await fetch(`/api/next?session_id=${sessionId}`);
        const data = await res.json();
        
        if (data.done) {
            showDone(data.progress?.total || 0);
            return;
        }
        
        currentImage = data;
        updateProgress(data.progress);
        renderImage(data);
        loadGarbageRating(data.id);
        preloadNext();
    } catch (err) {
        console.error('Failed to load next image:', err);
        showError('Failed to load image. Check server connection.');
    }
}

async function loadGarbageRating(imageId) {
    try {
        const res = await fetch(`/api/garbage-rating/${imageId}`);
        const data = await res.json();
        
        const ratingEl = document.getElementById('garbage-rating');
        if (ratingEl && data.garbage_score !== undefined) {
            const score = data.garbage_score;
            const emoji = score > 0.7 ? '🚮' : score > 0.4 ? '⚠️' : '✨';
            const label = data.quality === 'good' ? 'Good' : data.quality === 'garbage' ? 'Garbage' : 'Unknown';
            ratingEl.innerHTML = `${emoji} Quality: ${label} (${data.confidence}% confidence)`;
        }
    } catch (err) {
        console.warn('Failed to load garbage rating:', err);
    }
}

async function preloadNext() {
    try {
        const res = await fetch('/api/batch?count=5');
        const data = await res.json();
        preloadedImages = data.images || [];
        
        // Preload images into browser cache
        preloadedImages.forEach(img => {
            const preload = new Image();
            preload.src = `/api/image/${img.id}`;
        });
    } catch (err) {
        console.warn('Preload failed:', err);
    }
}

async function label(style, qualityFlag = null) {
    if (!currentImage || isLabeling) return;
    
    isLabeling = true;
    
    try {
        // Handle BAD_QUALITY case - it's actually a REFUSE with a flag
        if (style === 'BAD_QUALITY') {
            style = 'REFUSE';
            qualityFlag = 'BAD_QUALITY';
        }
        
        const payload = {
            image_id: currentImage.id,
            label: style,
            session_id: sessionId
        };
        
        // Add quality_flag if present
        if (qualityFlag) {
            payload.quality_flag = qualityFlag;
        }
        
        const res = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        if (data.success) {
            updateProgress(data.progress);
            
            // Quick visual feedback
            flashButton(style);
            
            // Both REFUSE and BAD_QUALITY get replacement from same cluster
            if (style === 'REFUSE') {
                loadReplacement(currentImage.cluster_id);
            } else {
                // Load next immediately for normal labels
                loadNext();
            }
        } else {
            showToast('Failed to save label');
        }
    } catch (err) {
        console.error('Label failed:', err);
        showToast('Error saving label');
    } finally {
        isLabeling = false;
    }
}

async function loadReplacement(clusterId) {
    try {
        const res = await fetch(`/api/replacement/${clusterId}`);
        const data = await res.json();
        
        if (data.done) {
            showDone(data.total || 993);
            return;
        }
        
        currentImage = data;
        updateProgress(data.progress);
        renderImage(data);
        loadGarbageRating(data.id);
        preloadNext();
    } catch (err) {
        console.error('Failed to load replacement:', err);
        // Fallback to random next image
        loadNext();
    }
}

async function undoLast() {
    try {
        const res = await fetch('/api/undo', { method: 'POST' });
        const data = await res.json();
        
        if (data.success) {
            showToast('Undone');
            loadNext();
        } else {
            showToast(data.message || 'Nothing to undo');
        }
    } catch (err) {
        console.error('Undo failed:', err);
        showToast('Undo failed');
    }
}

// ============================================================================
// Rendering
// ============================================================================

function renderImage(data) {
    const main = document.getElementById('main');
    
    const seriesDisplay = data.series_name ? `<span class="series-name">${data.series_name}</span>` : '';
    
    const metaInfo = [];
    if (data.production_year) metaInfo.push(`📅 ${data.production_year}`);
    if (data.demographic) metaInfo.push(`👥 ${capitalize(data.demographic)}`);
    const metaDisplay = metaInfo.length > 0 ? `<span class="meta-tags">${metaInfo.join(' | ')}</span>` : '';
    
    const clusterInfo = data.cluster_id !== undefined ? `<span class="cluster-info">Cluster ${data.cluster_id}</span>` : '';
    
    // Build prediction display if available
    let predictionHtml = '';
    if (data.predicted_style) {
        predictionHtml = `
            <button class="btn-reveal" onclick="togglePrediction()" title="Click to reveal model's prediction">
                🔍 Reveal prediction
            </button>
            <div id="predictionSpoiler" class="prediction-spoiler hidden">
                Model predicted: <strong>${data.predicted_style}</strong> 
                (${Math.round((data.predicted_confidence || 0) * 100)}%)
            </div>
        `;
    }
    
    // Build buttons dynamically from config
    const styleButtons = STYLES.map((s, i) => {
        const color = STYLE_COLORS[s] || getDefaultColor(i);
        return `
            <button class="btn" style="background:${color};color:white" onclick="label('${s}')" id="btn-${s}">
                ${capitalize(s)} <kbd>${i + 1}</kbd>
            </button>
        `;
    }).join('');
    
    main.innerHTML = `
        <div class="image-container">
            <img src="/api/image/${data.id}" alt="Frame to label" 
                 onload="this.style.opacity=1" style="opacity:0;transition:opacity 0.2s">
        </div>
        <div class="meta">
            ${seriesDisplay}
            ${metaDisplay}
            ${clusterInfo}
            <span id="garbage-rating" class="garbage-rating">⏳ Analyzing quality...</span>
            ${predictionHtml}
        </div>
        <div class="buttons">
            ${styleButtons}
            <button class="btn btn-refuse" onclick="label('REFUSE')" id="btn-REFUSE">
                Ambiguous <kbd>X</kbd>
            </button>
            <button class="btn btn-bad-quality" onclick="label('REFUSE', 'BAD_QUALITY')" id="btn-BAD_QUALITY">
                Bad Quality <kbd>Q</kbd>
            </button>
        </div>
    `;
}

function getDefaultColor(idx) {
    const colors = ["#607D8B", "#37474F", "#1976D2", "#C2185B", "#7B1FA2", "#F57C00", "#00897B", "#5D4037"];
    return colors[idx % colors.length];
}

function updateProgress(progress) {
    if (!progress) return;
    
    const fill = document.getElementById('progressFill');
    const text = document.getElementById('progressText');
    
    if (fill) fill.style.width = `${progress.percent}%`;
    if (text) text.textContent = `${progress.labeled} / ${progress.total} (${progress.percent}%)`;
}

function showDone(total) {
    const main = document.getElementById('main');
    main.innerHTML = `
        <div class="done">
            <h2>🎉 All Done!</h2>
            <p>You've labeled all ${total} images.</p>
            <p style="margin-top: 20px;">
                <a href="/api/export" download="labels.json">📥 Download Labels (JSON)</a>
            </p>
            <p style="margin-top: 10px;">
                <button class="btn-primary" onclick="showStats()">View Statistics</button>
            </p>
        </div>
    `;
}

function showError(message) {
    const main = document.getElementById('main');
    main.innerHTML = `
        <div class="done">
            <h2>❌ Error</h2>
            <p>${message}</p>
            <button class="btn-primary" onclick="loadNext()">Retry</button>
        </div>
    `;
}

function flashButton(style) {
    const btn = document.getElementById(`btn-${style}`);
    if (btn) {
        btn.style.transform = 'scale(1.1)';
        btn.style.filter = 'brightness(1.3)';
        setTimeout(() => {
            btn.style.transform = '';
            btn.style.filter = '';
        }, 150);
    }
}

function togglePrediction() {
    const spoiler = document.getElementById('predictionSpoiler');
    const btn = document.querySelector('.btn-reveal');
    if (spoiler.classList.contains('hidden')) {
        spoiler.classList.remove('hidden');
        btn.textContent = '🙈 Hide prediction';
    } else {
        spoiler.classList.add('hidden');
        btn.textContent = '🔍 Reveal prediction';
    }
}

// ============================================================================
// Statistics Modal
// ============================================================================

async function showStats() {
    const modal = document.getElementById('statsModal');
    const content = document.getElementById('statsContent');
    
    modal.classList.add('active');
    content.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        
        const allLabels = STYLES.concat(['REFUSE']);
        const maxCount = Math.max(...allLabels.map(l => stats.by_label?.[l] || 0), 1);
        
        content.innerHTML = `
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Total</div>
                    <div class="stat-value">${stats.total}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Labeled</div>
                    <div class="stat-value">${stats.labeled}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Remaining</div>
                    <div class="stat-value">${stats.remaining}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Progress</div>
                    <div class="stat-value">${Math.round(100 * stats.labeled / stats.total)}%</div>
                </div>
            </div>
            
            <h3>Labels Distribution</h3>
            <div class="label-bars">
                ${allLabels.map(s => {
                    const count = stats.by_label?.[s] || 0;
                    const pct = 100 * count / maxCount;
                    const color = STYLE_COLORS[s] || (s === 'REFUSE' ? '#c62828' : '#757575');
                    return `
                        <div class="label-bar">
                            <span class="name">${capitalize(s)}</span>
                            <div class="bar">
                                <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
                            </div>
                            <span class="count">${count}</span>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    } catch (err) {
        content.innerHTML = `<p>Failed to load statistics</p>`;
    }
}

function closeStats() {
    document.getElementById('statsModal').classList.remove('active');
}

function closeModal(event) {
    if (event.target.classList.contains('modal')) {
        event.target.classList.remove('active');
    }
}

// ============================================================================
// History Modal
// ============================================================================

let currentHistoryPage = 1;

async function showHistory() {
    const modal = document.getElementById('historyModal');
    const filter = document.getElementById('historyFilter');
    
    // Populate filter options from config
    filter.innerHTML = '<option value="">All Labels</option>' + 
        STYLES.concat(['REFUSE']).map(s => 
            `<option value="${s}">${capitalize(s)}</option>`
        ).join('');
    
    modal.classList.add('active');
    loadHistory(1);
}

async function loadHistory(page = 1) {
    currentHistoryPage = page;
    const content = document.getElementById('historyContent');
    const filter = document.getElementById('historyFilter').value;
    
    content.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const url = `/api/history?page=${page}&per_page=12${filter ? `&label_filter=${filter}` : ''}`;
        const res = await fetch(url);
        const data = await res.json();
        
        if (!data.items || data.items.length === 0) {
            content.innerHTML = '<p class="history-empty">No labeled items yet</p>';
            document.getElementById('historyPagination').innerHTML = '';
            return;
        }
        
        content.innerHTML = `
            <div class="history-grid">
                ${data.items.map(item => `
                    <div class="history-item" onclick="relabelFromHistory('${item.id}')">
                        <img src="/api/image/${item.id}" alt="Labeled frame" loading="lazy">
                        <div class="history-label" style="background:${STYLE_COLORS[item.label] || '#c62828'}">
                            ${capitalize(item.label)}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        // Pagination
        const pagination = document.getElementById('historyPagination');
        const totalPages = Math.ceil(data.total / data.per_page);
        
        if (totalPages > 1) {
            pagination.innerHTML = `
                <button class="btn-secondary" ${page <= 1 ? 'disabled' : ''} onclick="loadHistory(${page - 1})">← Prev</button>
                <span class="page-info">Page ${page} of ${totalPages}</span>
                <button class="btn-secondary" ${page >= totalPages ? 'disabled' : ''} onclick="loadHistory(${page + 1})">Next →</button>
            `;
        } else {
            pagination.innerHTML = '';
        }
        
    } catch (err) {
        console.error('Failed to load history:', err);
        content.innerHTML = '<p>Failed to load history</p>';
    }
}

async function relabelFromHistory(imageId) {
    const newLabel = prompt(`Enter new label for this image.\nValid: ${STYLES.join(', ')}, REFUSE`);
    
    if (!newLabel) return;
    
    const normalizedLabel = newLabel.toLowerCase() === 'refuse' ? 'REFUSE' : newLabel.toLowerCase();
    
    if (!STYLES.includes(normalizedLabel) && normalizedLabel !== 'REFUSE') {
        showToast(`Invalid label: ${newLabel}`);
        return;
    }
    
    try {
        const res = await fetch(`/api/history/${imageId}/relabel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: normalizedLabel })
        });
        
        const data = await res.json();
        
        if (data.success) {
            showToast(`Relabeled to ${normalizedLabel}`);
            loadHistory(currentHistoryPage);
        } else {
            showToast(data.message || 'Failed to relabel');
        }
    } catch (err) {
        console.error('Relabel failed:', err);
        showToast('Failed to relabel');
    }
}

function closeHistory() {
    document.getElementById('historyModal').classList.remove('active');
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(message) {
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// ============================================================================
// Utilities
// ============================================================================

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    const key = e.key.toLowerCase();
    
    // Undo
    if (key === 'z') {
        e.preventDefault();
        undoLast();
        return;
    }
    
    // History
    if (key === 'h') {
        e.preventDefault();
        showHistory();
        return;
    }
    
    // Escape closes modal
    if (key === 'escape') {
        closeStats();
        closeHistory();
        return;
    }
    
    // Style shortcuts
    if (KEY_MAP[key]) {
        e.preventDefault();
        label(KEY_MAP[key]);
    }
});

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    loadNext();
});

// Expose to global for onclick handlers
window.label = label;
window.undoLast = undoLast;
window.showStats = showStats;
window.closeStats = closeStats;
window.closeModal = closeModal;
window.showHistory = showHistory;
window.loadHistory = loadHistory;
window.closeHistory = closeHistory;
window.relabelFromHistory = relabelFromHistory;
