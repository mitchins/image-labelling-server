const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const appPath = path.join(root, 'static/app.js');
const mirroredAppPath = path.join(root, 'smart_label/static/app.js');
const htmlPath = path.join(root, 'static/index.html');
const mirroredHtmlPath = path.join(root, 'smart_label/static/index.html');
const cssPath = path.join(root, 'static/style.css');
const mirroredCssPath = path.join(root, 'smart_label/static/style.css');
const appSource = fs.readFileSync(appPath, 'utf8');
const htmlSource = fs.readFileSync(htmlPath, 'utf8');

function loadRequestHelpers() {
    let requestNumber = 0;
    const context = {
        console,
        crypto: { randomUUID: () => `request-${++requestNumber}` },
        document: { addEventListener() {} },
        window: {},
        localStorage: {
            getItem() { return null; },
            setItem() {}
        }
    };
    vm.runInNewContext(`${appSource}
        this.__test = { getRankingRequestId, clearPendingRankingRequest };`, context);
    return context.__test;
}

test('frontend copies remain byte-for-byte identical', () => {
    assert.equal(fs.readFileSync(appPath, 'utf8'), fs.readFileSync(mirroredAppPath, 'utf8'));
    assert.equal(fs.readFileSync(htmlPath, 'utf8'), fs.readFileSync(mirroredHtmlPath, 'utf8'));
    assert.equal(fs.readFileSync(cssPath, 'utf8'), fs.readFileSync(mirroredCssPath, 'utf8'));
});

test('ranking request IDs are operation and endpoint scoped and retry-stable', () => {
    const { getRankingRequestId, clearPendingRankingRequest } = loadRequestHelpers();
    const payload = { set_id: 'set-1', expected_revision: 2, outcome: 'ranked' };

    const submitId = getRankingRequestId(payload, 'submit', '/api/rank');
    const correctionId = getRankingRequestId(payload, 'history-correction', '/api/history/rerank');
    assert.notEqual(submitId, correctionId);
    assert.equal(getRankingRequestId(payload, 'submit', '/api/rank'), submitId);
    assert.equal(getRankingRequestId(payload, 'history-correction', '/api/history/rerank'), correctionId);

    clearPendingRankingRequest(payload, 'history-correction', '/api/history/rerank');
    assert.notEqual(getRankingRequestId(payload, 'history-correction', '/api/history/rerank'), correctionId);
    assert.equal(getRankingRequestId(payload, 'submit', '/api/rank'), submitId);
});

test('modal focus contract and history correction lifecycle are present', () => {
    assert.match(htmlSource, /id="statsModal"[^>]*tabindex="-1"/);
    assert.match(htmlSource, /id="historyModal"[^>]*tabindex="-1"/);
    assert.match(appSource, /modalOpeners\.set\(modal\.id, document\.activeElement\)/);
    assert.match(appSource, /function trapModalFocus\(event\)/);
    assert.match(appSource, /closeModalElement\(document\.querySelector\('\.modal\.active'\)\)/);
    assert.match(appSource, /state\.inFlight = true/);
    assert.match(appSource, /finally \{[\s\S]*setHistoryCorrectionBusy\(panel, false\)/);
});

test('audio autoplay is user-controlled and excluded from ranking candidates', () => {
    assert.match(htmlSource, /id="audioAutoplayToggle"[^>]*aria-pressed="false"[^>]*hidden/);
    assert.match(appSource, /audio_autoplay_persistence \|\| 'session'/);
    assert.match(appSource, /document\.cookie = `\$\{key\}=\$\{enabled \? '1' : '0'\}/);
    assert.match(appSource, /sessionStorage\.setItem\(key, enabled \? '1' : '0'\)/);
    assert.equal((appSource.match(/\$\{audioAutoplayAttribute\(\)\}/g) || []).length, 2);
    assert.doesNotMatch(appSource, /ranking-audio[^`]*audioAutoplayAttribute/);
});

test('space never labels, submits, or activates review cards', () => {
    assert.doesNotMatch(appSource, /KEY_MAP\[' '\]/);
    assert.doesNotMatch(appSource, /\['Enter', ' '\]/);
    assert.doesNotMatch(appSource, /event\.key !== 'Enter' && event\.key !== ' '/);
    assert.match(appSource, /function allowsNativeSpace\(target\)[\s\S]*closest\('audio, input, textarea, select, \[contenteditable="true"\]'\)/);
    assert.match(appSource, /if \(key === ' '\) \{[\s\S]*allowsNativeSpace\(e\.target\)[\s\S]*e\.preventDefault\(\);[\s\S]*return;/);
    assert.match(appSource, /addEventListener\('keyup'[\s\S]*e\.key === ' '[\s\S]*e\.preventDefault\(\)/);
});
