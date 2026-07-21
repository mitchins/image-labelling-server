/**
 * Smart Label - Application Logic
 * Zero-friction labeling with keyboard shortcuts and preloading
 */

// Utility: Generate UUID if crypto.randomUUID is not available
function generateUUID() {
    if (crypto && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// State
let currentItem = null;
let currentRankingSet = null;
let rankingDraft = [];
let rankingSubmitting = false;
let rankingFocusedCandidateId = null;
const pendingRankingRequests = new Map();
let pendingRankingUndo = null;
let rankingUndoInFlight = false;
const rankingSubmissionStack = [];
const historyCorrectionStates = new WeakMap();
let preloadedImages = [];
let isLabeling = false;
let lastSubmitAt = 0;
let sessionId = localStorage.getItem('smartLabelSession') || generateUUID();
localStorage.setItem('smartLabelSession', sessionId);

// Config (loaded from server)
let CONFIG = null;
let STYLES = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro'];
let STYLE_COLORS = {};
let KEY_MAP = {};
let audioAutoplayEnabled = false;

function audioAutoplayPreferenceKey() {
    const task = String(CONFIG?.name || 'default').replace(/[^a-z0-9_-]+/gi, '_');
    return `smart_label_audio_autoplay_${task}`;
}

function readAudioAutoplayPreference() {
    const key = audioAutoplayPreferenceKey();
    const persistence = CONFIG?.audio_autoplay_persistence || 'session';
    const stored = persistence === 'cookie'
        ? document.cookie.split('; ').find((entry) => entry.startsWith(`${key}=`))?.split('=')[1]
        : sessionStorage.getItem(key);
    return stored === undefined || stored === null
        ? Boolean(CONFIG?.audio_autoplay_default)
        : stored === '1';
}

function writeAudioAutoplayPreference(enabled) {
    const key = audioAutoplayPreferenceKey();
    if ((CONFIG?.audio_autoplay_persistence || 'session') === 'cookie') {
        document.cookie = `${key}=${enabled ? '1' : '0'}; Path=/; Max-Age=31536000; SameSite=Lax`;
    } else {
        sessionStorage.setItem(key, enabled ? '1' : '0');
    }
}

function updateAudioAutoplayControl(mediaType = CONFIG?.media_type) {
    const button = document.getElementById('audioAutoplayToggle');
    if (!button) return;
    const available = !isRankingMode() && String(mediaType || '').toLowerCase() === 'audio';
    button.hidden = !available;
    button.textContent = `Autoplay: ${audioAutoplayEnabled ? 'On' : 'Off'}`;
    button.setAttribute('aria-pressed', String(audioAutoplayEnabled));
}

function toggleAudioAutoplay() {
    if (isRankingMode()) return;
    audioAutoplayEnabled = !audioAutoplayEnabled;
    writeAudioAutoplayPreference(audioAutoplayEnabled);
    updateAudioAutoplayControl(getTaskMediaType());
    if (audioAutoplayEnabled) document.getElementById('currentAudio')?.play().catch(() => {});
}

function audioAutoplayAttribute() {
    return !isRankingMode() && audioAutoplayEnabled ? ' autoplay' : '';
}

function isConfirmationMode() {
    return CONFIG?.mode === 'ontology_confirmation';
}

function isRankingMode() {
    return CONFIG?.mode === 'ranking';
}

function getRankingSetId(set = currentRankingSet) {
    return set?.set_id ?? set?.id;
}

function getCandidateId(candidate) {
    return candidate?.candidate_id ?? candidate?.id;
}

function getCandidateDisplayPosition(candidate, fallbackPosition = 1) {
    const position = Number(candidate?.display_position);
    return Number.isFinite(position) && position > 0 ? position : fallbackPosition;
}

function getRankingCandidates(set = currentRankingSet) {
    return Array.isArray(set?.candidates)
        ? set.candidates.slice().sort((a, b) => getCandidateDisplayPosition(a) - getCandidateDisplayPosition(b))
        : [];
}

function rankingPayloadKey(payload) {
    return JSON.stringify(payload);
}

function rankingRequestStateKey(payload, operation, endpoint) {
    return `${operation}:${endpoint}:${rankingPayloadKey(payload)}`;
}

function getRankingRequestId(payload, operation = 'submit', endpoint = '/api/rank') {
    const key = rankingPayloadKey(payload);
    const stateKey = rankingRequestStateKey(payload, operation, endpoint);
    const pending = pendingRankingRequests.get(stateKey);
    if (!pending || pending.key !== key) {
        pendingRankingRequests.set(stateKey, { key, requestId: generateUUID() });
    }
    return pendingRankingRequests.get(stateKey).requestId;
}

function clearPendingRankingRequest(payload = null, operation = 'submit', endpoint = '/api/rank') {
    const scope = `${operation}:${endpoint}:`;
    if (!payload) {
        for (const stateKey of pendingRankingRequests.keys()) {
            if (stateKey.startsWith(scope)) pendingRankingRequests.delete(stateKey);
        }
        return;
    }
    const stateKey = rankingRequestStateKey(payload, operation, endpoint);
    if (pendingRankingRequests.get(stateKey)?.key === rankingPayloadKey(payload)) {
        pendingRankingRequests.delete(stateKey);
    }
}

function isDefinitiveRankingResponse(res) {
    return res.ok || [400, 409, 422].includes(res.status);
}

function getCandidateMediaType(candidate) {
    if (candidate?.media_type) return String(candidate.media_type).toLowerCase();
    const path = String(candidate?.path || candidate?.url || '').toLowerCase();
    return /\.(mp3|wav|ogg|m4a|flac|aac|opus)(\?.*)?$/.test(path) ? 'audio' : 'image';
}

function getCandidateMediaUrl(candidate) {
    if (candidate?.url) return candidate.url;
    if (candidate?.id !== undefined && candidate?.id !== null) {
        return `/api/media/${encodeURIComponent(candidate.id)}`;
    }
    return candidate?.path || '';
}

function getTaskMediaType(item = currentItem) {
    return (item?.media_type || CONFIG?.media_type || 'image').toLowerCase();
}

function getMediaUrl(item = currentItem) {
    return `/api/media/${item.id}`;
}

function getItemName(item = currentItem) {
    return escapeHtml((item?.path || '').split(/[\\/]/).pop() || `${capitalize(getTaskMediaType(item))} item`);
}

function formatMetadataValue(value) {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        CONFIG = await res.json();
        audioAutoplayEnabled = readAudioAutoplayPreference();

        STYLES = CONFIG.labels || STYLES;
        STYLE_COLORS = CONFIG.label_colors || {};

        KEY_MAP = {};
        STYLES.forEach((s, i) => {
            if (i < 9) KEY_MAP[String(i + 1)] = s;
        });
        KEY_MAP.x = 'REFUSE';
        KEY_MAP.q = 'BAD_QUALITY';
        KEY_MAP.r = 'REPLAY';

        document.getElementById('taskName').textContent = CONFIG.name || 'Labeling Task';
        updateAudioAutoplayControl();
        updateShortcutsDisplay();
    } catch (err) {
        console.warn('Failed to load config, using defaults:', err);
    }
}

function updateShortcutsDisplay(mediaType = getTaskMediaType()) {
    const shortcuts = document.getElementById('shortcuts');
    if (!shortcuts) return;

    if (isConfirmationMode()) {
        shortcuts.innerHTML = ['1 STRONG', '2 LOOSE', '3 NONE', '4 INVALID', 'R Replay', 'Z Undo', 'H History']
            .map((label) => `<span class="shortcut-group"><kbd>${escapeHtml(label.split(' ')[0])}</kbd> ${escapeHtml(label.slice(2))}</span>`).join('');
        return;
    }

    if (isRankingMode()) {
        const count = getRankingCandidates().length;
        const numberHint = count === 2 ? '1/2 Winner first' : '1-8 Add to order, Enter Submit, Backspace Remove';
        shortcuts.innerHTML = `<span class="shortcut-group"><kbd>${numberHint.split(' ')[0]}</kbd> ${escapeHtml(numberHint.slice(numberHint.indexOf(' ') + 1))}</span>` +
            ' <span class="shortcut-group"><kbd>X</kbd> Invalid set</span>' +
            (getRankingCandidates().some((candidate) => getCandidateMediaType(candidate) === 'audio') ? ' <span class="shortcut-group"><kbd>R</kbd> Replay</span>' : '') +
            ' <span class="shortcut-group"><kbd>Z</kbd> Undo</span>' +
            ' <span class="shortcut-group"><kbd>H</kbd> History</span>';
        return;
    }

    const groups = STYLES.map((s, i) =>
        `<span class="shortcut-group"><kbd>${i + 1}</kbd> ${capitalize(s)}</span>`
    ).join('');

    shortcuts.innerHTML = groups +
        ' <span class="shortcut-group"><kbd>X</kbd> Refuse</span>' +
        (mediaType === 'image' ? ' <span class="shortcut-group"><kbd>Q</kbd> Bad Quality</span>' : '') +
        (mediaType === 'audio' ? ' <span class="shortcut-group"><kbd>R</kbd> Replay</span>' : '') +
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

        if (isRankingMode()) {
            currentRankingSet = data.set || null;
            currentItem = data;
            rankingDraft = [];
            clearPendingRankingRequest();
            rankingSubmitting = false;
            rankingFocusedCandidateId = null;
            updateProgress(data.progress);
            renderItem(data);
            return;
        }

        currentItem = data;
        updateProgress(data.progress);
        renderItem(data);
        if (getTaskMediaType(data) === 'image') {
            loadGarbageRating(data.id);
        }
        preloadNext();
    } catch (err) {
        console.error('Failed to load next item:', err);
        showError('Failed to load item. Check server connection.');
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
        preloadedImages = (data.images || []).filter((item) => getTaskMediaType(item) === 'image');

        preloadedImages.forEach((img) => {
            const preload = new Image();
            preload.src = getMediaUrl(img);
        });
    } catch (err) {
        console.warn('Preload failed:', err);
    }
}

async function label(style, qualityFlag = null) {
    if (!currentItem || isLabeling || Date.now() - lastSubmitAt < 250) return;
    if (qualityFlag === 'BAD_QUALITY' && getTaskMediaType() !== 'image') return;

    isLabeling = true;
    lastSubmitAt = Date.now();
    setLabelingBusy(true);

    try {
        if (style === 'BAD_QUALITY') {
            style = 'REFUSE';
            qualityFlag = 'BAD_QUALITY';
        }

        const payload = {
            image_id: currentItem.id,
            label: style,
            session_id: sessionId
        };

        if (qualityFlag) {
            payload.quality_flag = qualityFlag;
        }

        const res = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (res.status === 409) {
            showToast('Already labeled elsewhere, loading next');
            loadNext();
            return;
        }

        if (data.success) {
            updateProgress(data.progress);
            flashButton(style);

            if (style === 'REFUSE') {
                loadReplacement(currentItem.cluster_id);
            } else {
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
        setLabelingBusy(false);
    }
}

async function confirmItem(confirmation, item = currentItem) {
    if (!item || isLabeling || Date.now() - lastSubmitAt < 250) return;
    isLabeling = true;
    lastSubmitAt = Date.now();
    setLabelingBusy(true);

    try {
        const res = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: item.id, confirmation, session_id: sessionId })
        });
        const data = await res.json();
        if (res.status === 409) {
            showToast('Already confirmed elsewhere, loading next');
            await loadNext();
            return;
        }
        if (!data.success) {
            showToast(data.message || 'Failed to save confirmation');
            return;
        }
        updateProgress(data.progress);
        renderReceipt(item, confirmation);
        flashButton(`confirmation-${confirmation}`);
        await loadNext();
    } catch (err) {
        console.error('Confirmation failed:', err);
        showToast('Error saving confirmation');
    } finally {
        isLabeling = false;
        setLabelingBusy(false);
    }
}

function renderReceipt(item, confirmation) {
    const receipt = document.getElementById('lastDecisionGlobal');
    if (receipt) receipt.innerHTML = `Last decision: <strong>${escapeHtml(item.indicative_value)}</strong> &rarr; <strong>${escapeHtml(confirmation)}</strong>`;
}

async function loadReplacement(clusterId) {
    try {
        const res = await fetch(`/api/replacement/${clusterId}`);
        const data = await res.json();

        if (data.done) {
            showDone(data.total || 993);
            return;
        }

        currentItem = data;
        updateProgress(data.progress);
        renderItem(data);
        if (getTaskMediaType(data) === 'image') {
            loadGarbageRating(data.id);
        }
        preloadNext();
    } catch (err) {
        console.error('Failed to load replacement:', err);
        loadNext();
    }
}

async function undoLast() {
    if (rankingUndoInFlight) return;
    if (isRankingMode() && rankingDraft.length) {
        showToast('Finish or clear the draft before persisted undo');
        return;
    }

    rankingUndoInFlight = true;
    try {
        const target = isRankingMode() ? rankingSubmissionStack.at(-1) : null;
        if (isRankingMode() && !target) {
            showToast('Nothing from this browser session to undo');
            return;
        }
        const targetKey = target ? `${target.set_id}:${target.revision}` : null;
        if (target && pendingRankingUndo?.key !== targetKey) {
            pendingRankingUndo = { key: targetKey, requestId: generateUUID() };
        }
        const undoPayload = target ? {
            request_id: pendingRankingUndo.requestId,
            expected_set_id: target.set_id,
            expected_revision: target.expected_revision,
            target_revision: target.revision,
            session_id: sessionId
        } : null;
        const res = await fetch(
            target ? '/api/undo' : `/api/undo?session_id=${encodeURIComponent(sessionId)}`,
            {
                method: 'POST',
                headers: target ? { 'Content-Type': 'application/json' } : {},
                body: target ? JSON.stringify(undoPayload) : undefined
            }
        );
        const data = await res.json();

        if (data.success) {
            if (target) {
                rankingSubmissionStack.pop();
                rankingSubmissionStack.forEach((entry) => {
                    if (entry.set_id === target.set_id) entry.expected_revision = data.revision.revision;
                });
                pendingRankingUndo = null;
            }
            showToast('Undone');
            if (isConfirmationMode()) {
                const item = data.item || data.returned_item;
                if (item && item.id !== undefined) {
                    const receipt = document.getElementById('lastDecisionGlobal');
                    if (receipt) receipt.innerHTML = `Undid decision for <strong>${escapeHtml(item.indicative_value)}</strong>`;
                    currentItem = item;
                    updateProgress(item.progress);
                    renderItem(item);
                    return;
                }
            }
            loadNext();
        } else {
            showToast(data.message || 'Nothing to undo');
        }
    } catch (err) {
        console.error('Undo failed:', err);
        showToast('Undo failed');
    } finally {
        rankingUndoInFlight = false;
    }
}

// ============================================================================
// Rendering
// ============================================================================

function renderItem(data) {
    const main = document.getElementById('main');
    const mediaType = getTaskMediaType(data);
    updateAudioAutoplayControl(mediaType);
    updateShortcutsDisplay(mediaType);

    if (isRankingMode()) {
        renderRankingItem(data);
        return;
    }

    if (isConfirmationMode()) {
        renderConfirmationItem(data);
        return;
    }

    const seriesDisplay = data.series_name ? `<span class="series-name">${data.series_name}</span>` : '';

    const metaInfo = [];
    if (data.production_year) metaInfo.push(`📅 ${data.production_year}`);
    if (data.demographic) metaInfo.push(`👥 ${capitalize(data.demographic)}`);
    const metaDisplay = metaInfo.length > 0 ? `<span class="meta-tags">${metaInfo.join(' | ')}</span>` : '';

    const clusterInfo = data.cluster_id !== undefined && data.cluster_id !== null
        ? `<span class="cluster-info">Cluster ${data.cluster_id}</span>`
        : '';

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

    const styleButtons = STYLES.map((s, i) => {
        const color = STYLE_COLORS[s] || getDefaultColor(i);
        return `
            <button class="btn" style="background:${color};color:white" onclick="label('${s}')" id="btn-${s}" data-label-action="true">
                ${capitalize(s)} <kbd>${i + 1}</kbd>
            </button>
        `;
    }).join('');

    const mediaHtml = mediaType === 'audio'
        ? `
            <div class="audio-review" aria-label="Audio review card">
                <div class="audio-title">🎧 ${getItemName(data)}</div>
                <audio id="currentAudio" src="${getMediaUrl(data)}" controls preload="auto"${audioAutoplayAttribute()}></audio>
                <button class="btn-secondary replay-btn" onclick="replayCurrentAudio()">Replay <kbd>R</kbd></button>
            </div>
        `
        : `
            <div class="image-container">
                <img src="${getMediaUrl(data)}" alt="Frame to label"
                     onload="this.style.opacity=1" style="opacity:0;transition:opacity 0.2s">
            </div>
        `;

    const qualityControls = mediaType === 'image'
        ? `
            <button class="btn btn-bad-quality" onclick="label('REFUSE', 'BAD_QUALITY')" id="btn-BAD_QUALITY" data-label-action="true">
                Bad Quality <kbd>Q</kbd>
            </button>
        `
        : '';

    const qualityMeta = mediaType === 'image'
        ? '<span id="garbage-rating" class="garbage-rating">⏳ Analyzing quality...</span>'
        : '<span class="garbage-rating">Audio review</span>';

    main.innerHTML = `
        ${mediaHtml}
        <div class="meta">
            ${seriesDisplay}
            ${metaDisplay}
            ${clusterInfo}
            ${qualityMeta}
            ${predictionHtml}
        </div>
        <div class="buttons">
            ${styleButtons}
            <button class="btn btn-refuse" onclick="label('REFUSE')" id="btn-REFUSE" data-label-action="true">
                Ambiguous <kbd>X</kbd>
            </button>
            ${qualityControls}
        </div>
    `;
}

function getCandidateName(candidate) {
    const path = String(candidate?.path || '').split(/[\\/]/).pop();
    return path || `Candidate ${getCandidateDisplayPosition(candidate)}`;
}

function renderCandidateMetadata(candidate) {
    if (!candidate?.metadata || typeof candidate.metadata !== 'object') return '';
    const entries = Object.entries(candidate.metadata);
    if (!entries.length) return '';
    return `<dl class="ranking-metadata">${entries.map(([key, value]) => `
        <div class="ranking-metadata-row"><dt>${escapeHtml(key.replaceAll('_', ' '))}</dt><dd>${escapeHtml(formatMetadataValue(value))}</dd></div>
    `).join('')}</dl>`;
}

function renderRankingCandidate(candidate, fallbackPosition) {
    const displayPosition = getCandidateDisplayPosition(candidate, fallbackPosition);
    const candidateId = String(getCandidateId(candidate));
    const mediaType = getCandidateMediaType(candidate);
    const mediaUrl = escapeHtml(getCandidateMediaUrl(candidate));
    const name = escapeHtml(getCandidateName(candidate));
    const isSelected = rankingDraft.includes(candidateId);
    const media = mediaType === 'audio'
        ? `<audio id="ranking-audio-${displayPosition}" src="${mediaUrl}" controls preload="metadata" aria-label="Audio for candidate ${displayPosition}"></audio>`
        : `<img src="${mediaUrl}" alt="${name}" loading="eager" onload="this.style.opacity=1" style="opacity:0;transition:opacity 0.2s">`;

    return `
        <article class="ranking-candidate ${isSelected ? 'is-selected' : ''}"
                 data-ranking-candidate data-candidate-id="${escapeHtml(candidateId)}"
                 data-display-position="${displayPosition}" tabindex="0" role="button"
                 aria-pressed="${isSelected}" aria-label="Candidate ${displayPosition}, ${name}">
            <div class="ranking-card-header">
                <span class="ranking-card-number" aria-hidden="true">${displayPosition}</span>
                <span class="ranking-card-label">${name}</span>
                ${isSelected ? '<span class="ranking-selected-mark" aria-label="Added to draft">✓</span>' : ''}
            </div>
            <div class="ranking-media ${mediaType === 'audio' ? 'ranking-media-audio' : ''}">${media}</div>
            ${renderCandidateMetadata(candidate)}
            <div class="ranking-card-hint">${isSelected ? `Rank ${rankingDraft.indexOf(candidateId) + 1}` : 'Select to add'}</div>
        </article>
    `;
}

function renderRankingDraft() {
    const candidates = getRankingCandidates();
    const byId = new Map(candidates.map((candidate) => [String(getCandidateId(candidate)), candidate]));
    if (!rankingDraft.length) {
        return '<p class="ranking-draft-empty">No candidates selected yet. Choose cards in best-first order.</p>';
    }

    return `<ol class="ranking-draft-list" aria-label="Draft ranking">${rankingDraft.map((candidateId, index) => {
        const candidate = byId.get(String(candidateId));
        if (!candidate) return '';
        const displayPosition = getCandidateDisplayPosition(candidate, index + 1);
        return `<li class="ranking-draft-item" draggable="true" data-draft-candidate-id="${escapeHtml(String(candidateId))}">
            <span class="ranking-draft-rank">${index + 1}</span>
            <span class="ranking-draft-name">Card ${displayPosition}: ${escapeHtml(getCandidateName(candidate))}</span>
            <button type="button" class="btn-secondary ranking-move" data-label-action="true" data-move-draft="up" data-candidate-id="${escapeHtml(String(candidateId))}" aria-label="Move card ${displayPosition} up" ${index === 0 ? 'disabled' : ''}>↑</button>
            <button type="button" class="btn-secondary ranking-move" data-label-action="true" data-move-draft="down" data-candidate-id="${escapeHtml(String(candidateId))}" aria-label="Move card ${displayPosition} down" ${index === rankingDraft.length - 1 ? 'disabled' : ''}>↓</button>
        </li>`;
    }).join('')}</ol>`;
}

function renderRankingItem(data) {
    const main = document.getElementById('main');
    const set = data?.set || currentRankingSet;
    if (!set || !Array.isArray(set.candidates)) {
        showError('Ranking set is missing or invalid.');
        return;
    }

    currentRankingSet = set;
    const candidates = getRankingCandidates(set);
    const criterion = set.criterion || CONFIG?.ranking_criterion || {};
    const count = candidates.length;
    const orderHint = count === 2
        ? 'Choose the winner first. The other card will follow automatically.'
        : 'Choose every card in strict best-to-worst order.';

    main.innerHTML = `
        <section class="ranking-shell" aria-labelledby="ranking-prompt">
            <div class="ranking-header">
                <div>
                    <div class="ranking-eyebrow">Ranking set${set.external_id ? ` · ${escapeHtml(set.external_id)}` : ''}</div>
                    <h2 id="ranking-prompt" class="ranking-prompt">${escapeHtml(criterion.prompt || 'Rank these candidates')}</h2>
                </div>
                <div class="ranking-direction" aria-label="Criterion direction">Most preferred first</div>
            </div>
            <div class="ranking-criterion-meta">Criterion ${escapeHtml(criterion.id || 'unknown')} · Version ${escapeHtml(criterion.version || 'unknown')}</div>
            <p class="ranking-instruction">${orderHint} <span class="ranking-key-hint">Card numbers are fixed display positions.</span></p>
            <div class="ranking-candidates" role="list" aria-label="Ranking candidates">
                ${candidates.map((candidate, index) => renderRankingCandidate(candidate, index + 1)).join('')}
            </div>
            <div class="ranking-draft-panel">
                <div class="ranking-draft-header">
                    <h3>Draft order <span class="ranking-draft-count">${rankingDraft.length}/${count}</span></h3>
                    <span class="ranking-drag-hint">Drag or use arrows to reorder</span>
                </div>
                <div id="rankingDraft" class="ranking-draft">${renderRankingDraft()}</div>
                <div class="ranking-actions">
                    <button type="button" class="btn btn-ranking-submit" data-ranking-submit data-label-action="true" ${count === 2 || rankingDraft.length !== count ? 'disabled' : ''}>Submit ranking <kbd>Enter</kbd></button>
                    <button type="button" class="btn btn-ranking-invalid" data-ranking-invalid data-label-action="true">Invalid set <kbd>X</kbd></button>
                </div>
            </div>
        </section>
    `;
    bindRankingControls();
}

function bindRankingControls() {
    document.querySelectorAll('[data-ranking-candidate]').forEach((card) => {
        card.addEventListener('focus', () => {
            rankingFocusedCandidateId = card.dataset.candidateId;
        });
        card.addEventListener('click', (event) => {
            if (event.target.closest('audio, button, a, input, select, textarea')) return;
            chooseRankingCandidate(card.dataset.candidateId);
        });
        card.addEventListener('keydown', (event) => {
            if (event.target !== card || event.key !== 'Enter') return;
            event.preventDefault();
            chooseRankingCandidate(card.dataset.candidateId);
        });
    });

    document.querySelector('[data-ranking-submit]')?.addEventListener('click', () => submitRanking());
    document.querySelector('[data-ranking-invalid]')?.addEventListener('click', () => submitRanking('invalid', [], 'user_marked_invalid'));

    document.querySelectorAll('[data-move-draft]').forEach((button) => {
        button.addEventListener('click', () => moveDraftCandidate(button.dataset.candidateId, button.dataset.moveDraft));
    });

    const draft = document.getElementById('rankingDraft');
    if (!draft) return;
    draft.addEventListener('dragstart', (event) => {
        const item = event.target.closest('[data-draft-candidate-id]');
        if (!item) return;
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', item.dataset.draftCandidateId);
        item.classList.add('is-dragging');
    });
    draft.addEventListener('dragend', (event) => event.target.closest('[data-draft-candidate-id]')?.classList.remove('is-dragging'));
    draft.addEventListener('dragover', (event) => {
        if (event.target.closest('[data-draft-candidate-id]')) event.preventDefault();
    });
    draft.addEventListener('drop', (event) => {
        const target = event.target.closest('[data-draft-candidate-id]');
        if (!target) return;
        event.preventDefault();
        const draggedCandidateId = event.dataTransfer.getData('text/plain');
        reorderDraftByCandidateId(draggedCandidateId, target.dataset.draftCandidateId);
    });
}

function chooseRankingCandidate(candidateId) {
    if (!isRankingMode() || rankingSubmitting || !currentRankingSet) return;
    const normalizedId = String(candidateId);
    const candidates = getRankingCandidates();
    if (!candidates.some((candidate) => String(getCandidateId(candidate)) === normalizedId)) return;

    if (candidates.length === 2) {
        rankingDraft = [normalizedId, ...candidates
            .map((candidate) => String(getCandidateId(candidate)))
            .filter((id) => id !== normalizedId)];
        clearPendingRankingRequest();
        renderRankingItem({ set: currentRankingSet });
        submitRanking();
        return;
    }

    if (rankingDraft.includes(normalizedId)) return;
    rankingDraft = [...rankingDraft, normalizedId];
    clearPendingRankingRequest();
    renderRankingItem({ set: currentRankingSet });
}

function moveDraftCandidate(candidateId, direction) {
    const index = rankingDraft.indexOf(String(candidateId));
    const nextIndex = direction === 'up' ? index - 1 : index + 1;
    if (index < 0 || nextIndex < 0 || nextIndex >= rankingDraft.length || rankingSubmitting) return;
    const nextDraft = rankingDraft.slice();
    [nextDraft[index], nextDraft[nextIndex]] = [nextDraft[nextIndex], nextDraft[index]];
    rankingDraft = nextDraft;
    clearPendingRankingRequest();
    renderRankingItem({ set: currentRankingSet });
}

function reorderDraftByCandidateId(draggedCandidateId, targetCandidateId) {
    const dragged = String(draggedCandidateId || '');
    const target = String(targetCandidateId || '');
    const fromIndex = rankingDraft.indexOf(dragged);
    const toIndex = rankingDraft.indexOf(target);
    if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex || rankingSubmitting) return;
    const nextDraft = rankingDraft.slice();
    nextDraft.splice(fromIndex, 1);
    nextDraft.splice(toIndex, 0, dragged);
    rankingDraft = nextDraft;
    clearPendingRankingRequest();
    renderRankingItem({ set: currentRankingSet });
}

function isCompleteRanking(candidateIds) {
    const expected = getRankingCandidates().map((candidate) => String(getCandidateId(candidate)));
    return candidateIds.length === expected.length && expected.every((id) => candidateIds.includes(id));
}

async function submitRanking(outcome = 'ranked', orderedCandidateIds = rankingDraft, invalidReason = null) {
    if (!isRankingMode() || !currentRankingSet || rankingSubmitting || isLabeling) return;
    const orderedIds = orderedCandidateIds.map((candidateId) => String(candidateId));
    if (outcome === 'ranked' && !isCompleteRanking(orderedIds)) {
        showToast('Choose every candidate before submitting');
        return;
    }

    rankingSubmitting = true;
    isLabeling = true;
    lastSubmitAt = Date.now();
    setLabelingBusy(true);
    const setId = getRankingSetId();
    const expectedRevision = currentRankingSet.revision;
    const logicalPayload = {
        set_id: setId,
        expected_revision: expectedRevision,
        session_id: sessionId,
        outcome,
        ordered_candidate_ids: outcome === 'ranked' ? orderedIds : [],
        invalid_reason: invalidReason || null
    };

    try {
        const payload = {
            ...logicalPayload,
            request_id: getRankingRequestId(logicalPayload, 'submit', '/api/rank')
        };

        const res = await fetch('/api/rank', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (isDefinitiveRankingResponse(res)) clearPendingRankingRequest(logicalPayload, 'submit', '/api/rank');

        if (res.status === 409) {
            showToast('Set changed elsewhere, loading the next set');
            await loadNext();
            return;
        }
        if (!res.ok || data.success === false) {
            showToast(data.message || data.detail || 'Failed to save ranking');
            return;
        }

        updateProgress(data.progress);
        rankingSubmissionStack.push({
            set_id: setId,
            expected_revision: data.revision.revision,
            revision: data.revision.revision
        });
        renderRankingReceipt(currentRankingSet, outcome, orderedIds);
        await loadNext();
    } catch (err) {
        console.error('Ranking failed:', err);
        showToast('Network error saving ranking; retry the same submission');
    } finally {
        rankingSubmitting = false;
        isLabeling = false;
        setLabelingBusy(false);
    }
}

function renderRankingReceipt(set, outcome, orderedIds) {
    const receipt = document.getElementById('lastDecisionGlobal');
    if (!receipt) return;
    if (outcome === 'invalid') {
        receipt.innerHTML = `Last ranking: <strong>Set ${escapeHtml(getRankingSetId(set))}</strong> marked invalid`;
        return;
    }
    const names = new Map(getRankingCandidates(set).map((candidate) => [String(getCandidateId(candidate)), getCandidateName(candidate)]));
    receipt.innerHTML = `Last ranking: <strong>${orderedIds.map((id) => escapeHtml(names.get(id) || id)).join(' &rarr; ')}</strong>`;
}

function renderConfirmationItem(data) {
    const main = document.getElementById('main');
    const ontology = Array.isArray(CONFIG?.ontology) ? CONFIG.ontology : [];
    const ontologyEntry = ontology.find((entry) => entry.id === data.indicative_value);
    const displayName = ontologyEntry?.display_name || data.indicative_value;
    const mediaType = getTaskMediaType(data);
    const choices = [
        ['STRONG', 'Strong match'], ['LOOSE', 'Loose match'], ['NONE', 'No match'], ['INVALID', 'Invalid item']
    ];
    const mediaHtml = mediaType === 'audio'
        ? `<div class="audio-review">
            <div class="audio-title">${getItemName(data)}</div>
            <audio id="currentAudio" src="${getMediaUrl(data)}" controls preload="auto"${audioAutoplayAttribute()}></audio>
            <button class="btn-secondary replay-btn" onclick="replayCurrentAudio()">Replay <kbd>R</kbd></button>
          </div>`
        : `<div class="image-container"><img src="${getMediaUrl(data)}" alt="Item to confirm"></div>`;
    const metadata = (CONFIG?.metadata_fields || [])
        .filter((field) => data[field] !== undefined && data[field] !== null && data[field] !== '')
        .map((field) => `<div class="confirmation-meta-row"><dt>${escapeHtml(field.replaceAll('_', ' '))}</dt><dd>${escapeHtml(formatMetadataValue(data[field]))}</dd></div>`)
        .join('');
    const itemKind = mediaType === 'audio' ? 'clip' : 'item';
    main.innerHTML = `
        ${mediaHtml}
        <div class="confirmation-card">
            <div class="confirmation-question">Does this ${itemKind} express <strong>${escapeHtml(displayName)}</strong>?</div>
            <div class="indicative-value">${escapeHtml(data.indicative_value)}</div>
            ${metadata ? `<dl class="confirmation-meta">${metadata}</dl>` : ''}
            <div class="confirmation-buttons">${choices.map(([value, label], index) => `<button class="btn confirmation-btn confirmation-${value.toLowerCase()}" id="btn-confirmation-${value}" data-label-action="true" onclick="confirmItem('${value}')">${escapeHtml(label)} <kbd>${index + 1}</kbd></button>`).join('')}</div>
        </div>`;
}

function getDefaultColor(idx) {
    const colors = ['#607D8B', '#37474F', '#1976D2', '#C2185B', '#7B1FA2', '#F57C00', '#00897B', '#5D4037'];
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
            <p>You've labeled all ${total} items.</p>
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

function setLabelingBusy(isBusy) {
    document.querySelectorAll('[data-label-action="true"]').forEach((button) => {
        button.disabled = isBusy;
        button.classList.toggle('is-busy', isBusy);
    });
    document.querySelectorAll('.ranking-candidate').forEach((card) => {
        card.classList.toggle('is-busy', isBusy);
        card.setAttribute('aria-disabled', String(isBusy));
    });
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

const modalOpeners = new Map();

function getModalFocusableElements(modal) {
    return Array.from(modal.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), audio, [tabindex]:not([tabindex="-1"])'
    )).filter((element) => !element.hidden && element.getAttribute('aria-hidden') !== 'true');
}

function openModal(modal) {
    if (!modal) return;
    if (!modal.classList.contains('active')) {
        modalOpeners.set(modal.id, document.activeElement);
    }
    modal.classList.add('active');
    const [firstFocusable] = getModalFocusableElements(modal);
    (firstFocusable || modal).focus();
}

function closeModalElement(modal) {
    if (!modal?.classList.contains('active')) return;
    modal.classList.remove('active');
    const opener = modalOpeners.get(modal.id);
    modalOpeners.delete(modal.id);
    if (opener && opener.isConnected !== false && typeof opener.focus === 'function') {
        opener.focus();
    }
}

function trapModalFocus(event) {
    if (event.key !== 'Tab') return false;
    const modal = document.querySelector('.modal.active');
    if (!modal) return false;

    const focusable = getModalFocusableElements(modal);
    if (!focusable.length) {
        event.preventDefault();
        modal.focus();
        return true;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !modal.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !modal.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
    }
    return true;
}

async function showStats() {
    const modal = document.getElementById('statsModal');
    const content = document.getElementById('statsContent');

    openModal(modal);
    content.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();

        if (isRankingMode()) {
            content.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Sets</div><div class="stat-value">${stats.total_sets}</div></div>
                    <div class="stat-item"><div class="stat-label">Ranked</div><div class="stat-value">${stats.ranked_sets}</div></div>
                    <div class="stat-item"><div class="stat-label">Invalid</div><div class="stat-value">${stats.invalid_sets}</div></div>
                    <div class="stat-item"><div class="stat-label">Remaining</div><div class="stat-value">${stats.remaining_sets}</div></div>
                    <div class="stat-item"><div class="stat-label">Progress</div><div class="stat-value">${stats.percent}%</div></div>
                </div>`;
            return;
        }

        const allLabels = isConfirmationMode()
            ? ['STRONG', 'LOOSE', 'NONE', 'INVALID']
            : STYLES.concat(['REFUSE']);
        const maxCount = Math.max(...allLabels.map((l) => stats.by_label?.[l] || 0), 1);

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
                ${allLabels.map((s) => {
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
        content.innerHTML = '<p>Failed to load statistics</p>';
    }
}

function closeStats() {
    closeModalElement(document.getElementById('statsModal'));
}

function closeModal(event) {
    if (event.target.classList.contains('modal')) {
        closeModalElement(event.target);
    }
}

// ============================================================================
// History Modal
// ============================================================================

let currentHistoryPage = 1;

async function showHistory() {
    const modal = document.getElementById('historyModal');
    const filter = document.getElementById('historyFilter');

    if (isRankingMode()) {
        filter.innerHTML = '<option value="">All Rankings</option>';
    } else {
        const filterValues = isConfirmationMode() ? ['STRONG', 'LOOSE', 'NONE', 'INVALID'] : STYLES.concat(['REFUSE']);
        filter.innerHTML = `<option value="">All ${isConfirmationMode() ? 'Outcomes' : 'Labels'}</option>` +
            filterValues.map((s) =>
                `<option value="${s}">${capitalize(s)}</option>`
            ).join('');
    }

    openModal(modal);
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

        const hasAudio = data.items.some((item) => getTaskMediaType(item) === 'audio');
        content.innerHTML = `
            <div class="history-grid ${hasAudio ? 'history-grid-audio' : ''}">
                ${data.items.map((item) => renderHistoryItem(item)).join('')}
            </div>
        `;
        if (isRankingMode()) bindRankingHistoryControls();

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

function renderHistoryItem(item) {
    if (isRankingMode()) return renderRankingHistoryItem(item);

    if (isConfirmationMode()) {
        const indicative = item.indicative_value ?? item.indicative ?? '';
        const outcome = item.confirmation ?? item.outcome ?? item.label ?? '';
        const entry = (CONFIG?.ontology || []).find((value) => value.id === indicative);
        const media = getTaskMediaType(item) === 'audio'
            ? `<audio id="history-audio-${item.id}" controls preload="none" src="${getMediaUrl(item)}"></audio>`
            : `<img src="${getMediaUrl(item)}" alt="Reviewed item" loading="lazy">`;
        return `<div class="confirmation-history-item">
            <div class="confirmation-history-media">${media}</div>
            <div class="confirmation-history-summary"><strong>${escapeHtml(entry?.display_name || indicative)}</strong><span class="history-outcome">${escapeHtml(outcome)}</span></div>
            <div class="confirmation-history-actions">${['STRONG', 'LOOSE', 'NONE', 'INVALID'].map((value) => `<button class="btn-secondary" onclick="confirmHistory(${Number(item.id)}, '${value}')">${value}</button>`).join('')}</div>
        </div>`;
    }
    const mediaType = getTaskMediaType(item);
    const labelColor = STYLE_COLORS[item.label] || '#c62828';

    if (mediaType === 'audio') {
        return `
            <div class="history-audio-item">
                <div class="history-audio-main">
                    <div class="history-audio-head">
                        <div class="history-audio-name">${getItemName(item)}</div>
                        <div class="history-label history-label-inline" style="background:${labelColor}">
                            ${capitalize(item.label)}
                        </div>
                    </div>
                    <audio id="history-audio-${item.id}" controls preload="none" src="${getMediaUrl(item)}" onclick="event.stopPropagation()"></audio>
                    <div class="history-audio-actions">
                        <button class="btn-secondary" onclick="event.stopPropagation(); replayHistoryAudio('${item.id}')">Replay</button>
                        <button class="btn-secondary" onclick="event.stopPropagation(); relabelFromHistory('${item.id}')">Relabel</button>
                    </div>
                </div>
            </div>
        `;
    }

    return `
        <div class="history-item" onclick="relabelFromHistory('${item.id}')">
            <img src="${getMediaUrl(item)}" alt="Labeled frame" loading="lazy">
            <div class="history-label" style="background:${labelColor}">
                ${capitalize(item.label)}
            </div>
        </div>
    `;
}

function renderRankingHistoryItem(item) {
    const set = item?.set || item || {};
    const setId = item?.set_id ?? set.set_id ?? set.id ?? '';
    const candidates = Array.isArray(item?.candidates) ? item.candidates : (Array.isArray(set.candidates) ? set.candidates : []);
    const orderedIds = (item?.ordered_candidate_ids || item?.order || item?.ranking || []).map((candidateId) => String(candidateId));
    const candidateNames = new Map(candidates.map((candidate) => [String(getCandidateId(candidate)), getCandidateName(candidate)]));
    const visibleOrder = orderedIds.length ? orderedIds : candidates.map((candidate) => String(getCandidateId(candidate)));
    const orderText = visibleOrder.map((candidateId) => candidateNames.get(candidateId) || candidateId).join(' -> ');
    const outcome = item?.outcome || (item?.invalid_reason ? 'invalid' : 'ranked');
    const correctionOrder = outcome === 'ranked' && orderedIds.length ? orderedIds : [];
    return `<div class="ranking-history-item">
        <div class="ranking-history-summary">
            <strong>Set ${escapeHtml(setId)}</strong>
            <span class="history-outcome">${escapeHtml(outcome)}</span>
            <div class="ranking-history-order">${escapeHtml(outcome === 'invalid' ? (item.invalid_reason || 'Whole set marked invalid') : orderText || 'No order recorded')}</div>
            <div class="ranking-history-correction" data-ranking-history-correction
                 data-set-id="${escapeHtml(String(setId))}"
                 data-revision="${escapeHtml(String(item?.revision ?? set.revision ?? ''))}"
                 data-order="${escapeHtml(JSON.stringify(correctionOrder))}">
                <div>Choose candidates in best-first order:</div>
                <div class="ranking-history-candidates">
                    ${candidates.map((candidate, index) => renderRankingHistoryCandidate(candidate, index + 1)).join('')}
                </div>
                <div class="ranking-history-correction-order" data-ranking-history-order></div>
                <div class="ranking-history-correction-actions">
                    <button type="button" class="btn-secondary" data-ranking-history-save>Save ranking</button>
                    <button type="button" class="btn-secondary" data-ranking-history-invalid>Mark whole set invalid</button>
                </div>
            </div>
        </div>
    </div>`;
}

function renderRankingHistoryCandidate(candidate, fallbackPosition) {
    const candidateId = String(getCandidateId(candidate));
    const displayPosition = getCandidateDisplayPosition(candidate, fallbackPosition);
    const mediaType = getCandidateMediaType(candidate);
    const mediaUrl = escapeHtml(getCandidateMediaUrl(candidate));
    const name = escapeHtml(getCandidateName(candidate));
    const media = mediaType === 'audio'
        ? `<audio src="${mediaUrl}" controls preload="metadata" aria-label="Audio for candidate ${displayPosition}"></audio>`
        : `<img src="${mediaUrl}" alt="${name}" loading="lazy">`;

    return `<article class="ranking-candidate" data-ranking-history-candidate
            data-candidate-id="${escapeHtml(candidateId)}" tabindex="0" role="button"
            aria-pressed="false" aria-label="Candidate ${displayPosition}, ${name}">
        <div class="ranking-card-header">
            <span class="ranking-card-number" aria-hidden="true">${displayPosition}</span>
            <span class="ranking-card-label">${name}</span>
        </div>
        <div class="ranking-media ${mediaType === 'audio' ? 'ranking-media-audio' : ''}">${media}</div>
        ${renderCandidateMetadata(candidate)}
        <div class="ranking-card-hint">Select to add</div>
    </article>`;
}

function getHistoryCorrectionState(panel) {
    let state = historyCorrectionStates.get(panel);
    if (!state) {
        state = { inFlight: false, payload: null };
        historyCorrectionStates.set(panel, state);
    }
    return state;
}

function setHistoryCorrectionBusy(panel, isBusy) {
    const state = getHistoryCorrectionState(panel);
    const controls = Array.from(panel.querySelectorAll('[data-ranking-history-candidate], [data-ranking-history-save], [data-ranking-history-invalid]'));
    if (isBusy) {
        state.disabledControls = new Map(controls.map((control) => [control, control.disabled]));
    }
    panel.dataset.correctionInFlight = String(isBusy);
    controls.forEach((control) => {
        if ('disabled' in control) {
            control.disabled = isBusy
                ? true
                : state.disabledControls?.get(control) ?? false;
        }
        control.setAttribute('aria-disabled', String(isBusy));
        control.classList.toggle('is-busy', isBusy);
    });
}

function bindRankingHistoryControls() {
    document.querySelectorAll('[data-ranking-history-correction]').forEach((panel) => {
        let order;
        try {
            order = JSON.parse(panel.dataset.order || '[]').map((candidateId) => String(candidateId));
        } catch (err) {
            showToast('History ranking is malformed');
            return;
        }

        const candidateCards = Array.from(panel.querySelectorAll('[data-ranking-history-candidate]'));
        const candidateIds = candidateCards.map((card) => card.dataset.candidateId);
        const candidateNames = new Map(candidateCards.map((card) => [card.dataset.candidateId, card.querySelector('.ranking-card-label')?.textContent || card.textContent]));
        const state = getHistoryCorrectionState(panel);
        const update = () => {
            const knownIds = new Set(candidateIds);
            order = order.filter((candidateId, index) => knownIds.has(candidateId) && order.indexOf(candidateId) === index);
            panel.querySelector('[data-ranking-history-order]').textContent = order.length
                ? order.map((candidateId, index) => `${index + 1}. ${candidateNames.get(candidateId) || candidateId}`).join(' -> ')
                : 'No candidates selected';
            candidateCards.forEach((card) => {
                const selected = order.includes(card.dataset.candidateId);
                card.setAttribute('aria-pressed', String(selected));
                card.classList.toggle('is-selected', selected);
            });
            const complete = order.length === candidateIds.length && candidateIds.every((candidateId) => order.includes(candidateId));
            panel.querySelector('[data-ranking-history-save]').disabled = !complete;
        };

        candidateCards.forEach((card) => {
            card.addEventListener('click', (event) => {
                if (state.inFlight) return;
                if (event.target.closest('audio, a, input, select, textarea')) return;
                if (state.payload) {
                    clearPendingRankingRequest(state.payload, 'history-correction', '/api/history/rerank');
                    state.payload = null;
                }
                const candidateId = card.dataset.candidateId;
                if (order.includes(candidateId)) {
                    order = order.filter((id) => id !== candidateId);
                } else {
                    order = [...order, candidateId];
                }
                update();
            });
            card.addEventListener('keydown', (event) => {
                if (event.target.closest('audio, a, input, select, textarea')) return;
                if (event.key !== 'Enter') return;
                event.preventDefault();
                card.click();
            });
        });
        panel.querySelector('[data-ranking-history-save]').addEventListener('click', () => {
            if (state.inFlight) return;
            rerankHistorySet(panel.dataset.setId, panel.dataset.revision, order, 'ranked', null, candidateIds, panel);
        });
        panel.querySelector('[data-ranking-history-invalid]').addEventListener('click', () => {
            if (state.inFlight) return;
            rerankHistorySet(panel.dataset.setId, panel.dataset.revision, [], 'invalid', 'user_marked_invalid', candidateIds, panel);
        });
        update();
    });
}

async function rerankHistorySet(setId, revision, orderedCandidateIds, outcome = 'ranked', invalidReason = null, knownCandidateIds = [], panel = null) {
    const orderedIds = orderedCandidateIds.map((candidateId) => String(candidateId));
    if (outcome === 'ranked' && (orderedIds.length !== knownCandidateIds.length || new Set(orderedIds).size !== orderedIds.length || !knownCandidateIds.every((candidateId) => orderedIds.includes(candidateId)))) {
        showToast('Choose every candidate exactly once');
        return;
    }

    const logicalPayload = {
        set_id: String(setId),
        expected_revision: Number(revision),
        session_id: sessionId,
        outcome,
        ordered_candidate_ids: outcome === 'ranked' ? orderedIds : [],
        invalid_reason: invalidReason || null
    };
    const state = panel ? getHistoryCorrectionState(panel) : null;
    if (state?.inFlight) return;
    if (state) {
        state.inFlight = true;
        state.payload = logicalPayload;
        setHistoryCorrectionBusy(panel, true);
    }

    try {
        const res = await fetch('/api/history/rerank', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...logicalPayload,
                request_id: getRankingRequestId(logicalPayload, 'history-correction', '/api/history/rerank')
            })
        });
        const data = await res.json();
        if (isDefinitiveRankingResponse(res)) {
            clearPendingRankingRequest(logicalPayload, 'history-correction', '/api/history/rerank');
            if (state) state.payload = null;
        }
        if (!res.ok || data.success === false) {
            showToast(data.message || data.detail || 'Failed to update ranking');
            return;
        }
        rankingSubmissionStack.push({
            set_id: data.revision.set_id,
            expected_revision: data.revision.revision,
            revision: data.revision.revision
        });
        showToast('Ranking revision saved');
        loadHistory(currentHistoryPage);
    } catch (err) {
        console.error('Ranking history correction failed:', err);
        showToast('Network error updating ranking; retry the same correction');
    } finally {
        if (state) {
            state.inFlight = false;
            setHistoryCorrectionBusy(panel, false);
        }
    }
}

async function confirmHistory(imageId, confirmation) {
    try {
        const res = await fetch(`/api/history/${imageId}/relabel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirmation, session_id: sessionId })
        });
        const data = await res.json();
        if (!res.ok || !data.success) {
            showToast(data.detail || 'Failed to change confirmation');
            return;
        }
        const receipt = document.getElementById('lastDecisionGlobal');
        if (receipt) receipt.innerHTML = `History correction &rarr; <strong>${escapeHtml(confirmation)}</strong>`;
        showToast(`Changed to ${confirmation}`);
        loadHistory(currentHistoryPage);
    } catch (err) {
        console.error('Confirmation correction failed:', err);
        showToast('Failed to change confirmation');
    }
}

async function relabelFromHistory(imageId) {
    const newLabel = prompt(`Enter new label for this item.\nValid: ${STYLES.join(', ')}, REFUSE`);

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
    closeModalElement(document.getElementById('historyModal'));
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

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function replayCurrentAudio() {
    const focusedAudio = document.activeElement?.tagName === 'AUDIO' ? document.activeElement : null;
    const focusedCandidateAudio = rankingFocusedCandidateId
        ? Array.from(document.querySelectorAll('[data-ranking-candidate]'))
            .find((card) => card.dataset.candidateId === rankingFocusedCandidateId)?.querySelector('audio')
        : null;
    const audio = isRankingMode()
        ? (focusedAudio || focusedCandidateAudio || document.querySelector('.ranking-candidates audio'))
        : document.getElementById('currentAudio');
    if (!audio) return;
    audio.currentTime = 0;
    audio.play().catch(() => {});
}

function replayHistoryAudio(itemId) {
    const audio = document.getElementById(`history-audio-${itemId}`);
    if (!audio) return;
    audio.currentTime = 0;
    audio.play().catch(() => {});
}

function isShortcutSuppressed(event) {
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest('input, textarea, select, button, a, audio, [contenteditable="true"]')) return true;
    return Boolean(document.querySelector('.modal.active'));
}

function allowsNativeSpace(target) {
    return target instanceof Element
        && Boolean(target.closest('audio, input, textarea, select, [contenteditable="true"]'));
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();

    if (key === ' ') {
        if (allowsNativeSpace(e.target)) return;
        e.preventDefault();
        return;
    }

    if (key === 'escape') {
        closeModalElement(document.querySelector('.modal.active'));
        return;
    }
    if (trapModalFocus(e)) return;
    if (isShortcutSuppressed(e)) return;

    if (isRankingMode()) {
        if (key === 'z') {
            e.preventDefault();
            if (rankingDraft.length) {
                showToast('Finish or clear the draft before persisted undo');
            } else {
                undoLast();
            }
            return;
        }
        if (key === 'h') {
            e.preventDefault();
            showHistory();
            return;
        }
        if (key === 'r') {
            e.preventDefault();
            replayCurrentAudio();
            return;
        }
        if (key === 'x') {
            e.preventDefault();
            submitRanking('invalid', [], 'user_marked_invalid');
            return;
        }
        if (key === 'backspace') {
            e.preventDefault();
            if (rankingDraft.length && !rankingSubmitting) {
                rankingDraft = rankingDraft.slice(0, -1);
                clearPendingRankingRequest();
                renderRankingItem({ set: currentRankingSet });
            }
            return;
        }
        if (key === 'enter') {
            e.preventDefault();
            submitRanking();
            return;
        }
        if (/^[1-8]$/.test(key)) {
            e.preventDefault();
            const candidate = getRankingCandidates().find((entry, index) => getCandidateDisplayPosition(entry, index + 1) === Number(key));
            if (candidate) chooseRankingCandidate(getCandidateId(candidate));
        }
        return;
    }

    if (key === 'z') {
        e.preventDefault();
        undoLast();
        return;
    }

    if (key === 'h') {
        e.preventDefault();
        showHistory();
        return;
    }

    if (KEY_MAP[key] === 'REPLAY') {
        e.preventDefault();
        replayCurrentAudio();
        return;
    }

    if (isConfirmationMode() && ['1', '2', '3', '4'].includes(key)) {
        e.preventDefault();
        confirmItem(['STRONG', 'LOOSE', 'NONE', 'INVALID'][Number(key) - 1]);
        return;
    }

    if (isConfirmationMode()) return;

    if (KEY_MAP[key]) {
        if (KEY_MAP[key] === 'BAD_QUALITY' && getTaskMediaType() !== 'image') return;
        e.preventDefault();
        label(KEY_MAP[key]);
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key === ' ' && !allowsNativeSpace(e.target)) e.preventDefault();
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
window.replayCurrentAudio = replayCurrentAudio;
window.replayHistoryAudio = replayHistoryAudio;
window.confirmItem = confirmItem;
window.confirmHistory = confirmHistory;
