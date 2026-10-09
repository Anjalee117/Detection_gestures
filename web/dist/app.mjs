import {CaptureGate} from './core.mjs';

const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('preview'), context = canvas.getContext('2d');
const gate = new CaptureGate();
let stream = null, worker = null, workerReady = false, busy = false, running = false;
let generation = 0, animation = 0, lastFrame = -1, lastSent = 0, pendingSince = 0;
let sentence = '', voices = [], speaking = false;
const edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function renderRecognition() {
  const state = gate.snapshot();
  $('letter').textContent = state.letter || '—';
  $('recognition-status').textContent = state.status;
  $('recognition-status').classList.toggle('ready', state.ready);
  $('confidence').textContent = state.confidence ? `${Math.round(state.confidence * 100)}%` : '—';
  $('confidence-bar').value = state.confidence || 0;
  $('stability').textContent = `${state.frames} / 5`;
  document.querySelectorAll('.frame-dots i').forEach((dot, i) => dot.classList.toggle('active', i < state.frames));
  $('capture').disabled = !running || !state.ready;
}
function renderSentence() {
  $('sentence').textContent = sentence || 'Captured letters will appear here.';
  $('sentence').classList.toggle('empty', !sentence);
  $('count').textContent = `${sentence.length} character${sentence.length === 1 ? '' : 's'}`;
  for (const id of ['space', 'delete', 'clear']) $(id).disabled = !sentence;
  $('speak').disabled = !sentence.trim() || !('speechSynthesis' in window) || speaking;
}
function capture() {
  const letter = running ? gate.capture() : '';
  if (letter) { sentence += letter; renderSentence(); }
  renderRecognition(); return {captured: letter, sentence};
}
function stop(text = 'Camera stopped. Start it again whenever you are ready.') {
  generation++; running = false; busy = false; workerReady = false;
  cancelAnimationFrame(animation); worker?.terminate(); worker = null;
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  video.srcObject = null; context.clearRect(0, 0, canvas.width, canvas.height);
  gate.reset(); renderRecognition(); $('placeholder').hidden = false;
  $('start').disabled = false; $('start').textContent = 'Start camera'; $('stop').disabled = true;
  $('camera-state').textContent = 'Camera off'; message(text);
}
function draw(landmarks = []) {
  if (!running || video.readyState < 2) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  }
  context.setTransform(-1, 0, 0, 1, canvas.width, 0); context.drawImage(video, 0, 0);
  context.setTransform(1, 0, 0, 1, 0, 0);
  if (!landmarks.length) return;
  context.lineWidth = Math.max(2, canvas.width / 300); context.strokeStyle = '#4ce1ac';
  for (const [a, b] of edges) {
    context.beginPath(); context.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
    context.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height); context.stroke();
  }
  context.fillStyle = '#ffffff';
  for (const p of landmarks) { context.beginPath(); context.arc(p.x * canvas.width, p.y * canvas.height, canvas.width / 160, 0, Math.PI * 2); context.fill(); }
}
let overlay = [], overlayAt = 0;
async function tick(now) {
  if (!running) return;
  draw(now - overlayAt < 500 ? overlay : []); renderRecognition();
  if (workerReady && busy && now - pendingSince > 15000) {
    stop(); message('Recognition stopped responding. Choose Start camera to retry.', true); return;
  }
  if (workerReady && !busy && video.readyState >= 2 && video.currentTime !== lastFrame && now - lastSent >= 65) {
    busy = true; pendingSince = now; lastSent = now; lastFrame = video.currentTime;
    const token = generation;
    try {
      const bitmap = await createImageBitmap(video);
      if (!running || token !== generation || !worker) { bitmap.close(); return; }
      worker.postMessage({type: 'frame', bitmap, timestamp: now}, [bitmap]);
    } catch (error) {
      if (token === generation) { stop(); message(`Camera frame could not be processed: ${error.message}. Retry the camera.`, true); }
      return;
    }
  }
  if (running) animation = requestAnimationFrame(tick);
}
async function cameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const selected = $('camera').value, devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  $('camera').replaceChildren(new Option('Default camera', ''));
  devices.forEach((d, i) => $('camera').add(new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  if ([...$('camera').options].some(o => o.value === selected)) $('camera').value = selected;
}
function cameraError(error) {
  if (error.name === 'NotAllowedError') return 'Camera permission was denied. Allow camera access in your browser’s site settings, then retry.';
  if (error.name === 'NotFoundError') return 'No camera found. Connect a camera and retry.';
  if (error.name === 'NotReadableError') return 'Camera is busy or unavailable. Close other camera apps, then retry.';
  if (error.name === 'OverconstrainedError') return 'Selected camera is unavailable. Select Default camera and retry.';
  return `Could not start the camera: ${error.message || 'unknown error'}. Please use a current Chrome, Edge or Safari browser and retry.`;
}
async function start() {
  const selected = $('camera').value;
  stop(); const token = generation;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { message('Camera requires HTTPS or localhost and a browser with camera support.', true); return; }
  if (!window.Worker || !window.OffscreenCanvas || !window.createImageBitmap) { message('This browser does not support webcam recognition. Use a current Chrome, Edge or Safari browser.', true); return; }
  $('start').disabled = true; $('start').textContent = 'Starting…'; $('stop').disabled = false;
  message('Allow camera access when your browser asks.');
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      ...(selected ? {deviceId: {exact: selected}} : {facingMode: 'user'}), width: {ideal: 640}, height: {ideal: 480}
    }});
    if (token !== generation) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; video.srcObject = stream;
    // Keep the decoder active on mobile browsers; the video remains visually hidden.
    video.hidden = false;
    await video.play();
    if (token !== generation) return;
    running = true; lastFrame = -1; lastSent = 0; overlay = []; overlayAt = 0;
    $('placeholder').hidden = true; $('camera-state').textContent = 'Camera on · loading recognition';
    message('Camera is live. Loading recognition; the first start can take a few seconds.');
    acquired.getVideoTracks()[0].addEventListener('ended', () => { if (token === generation) { stop(); message('Camera disconnected. Reconnect it and choose Start camera.', true); } });
    worker = new Worker('./worker.mjs');
    const timeout = setTimeout(() => { if (token === generation && !workerReady) { stop(); message('Recognition took too long to load. Check your connection and retry.', true); } }, 90000);
    worker.onmessage = ({data}) => {
      if (token !== generation) return;
      if (data.type === 'ready') {
        clearTimeout(timeout); workerReady = true; $('camera-state').textContent = 'Camera on';
        $('start').disabled = false; $('start').textContent = 'Retry camera';
        message('Show one hand. Hold a static letter pose, then press Capture letter.');
      } else if (data.type === 'result') {
        busy = false;
        if (performance.now() - data.timestamp > 500) { gate.reset(); overlay = []; }
        else { gate.observe(data.letter, data.confidence); overlay = data.landmarks; overlayAt = performance.now(); }
        renderRecognition();
      } else if (data.type === 'error') { clearTimeout(timeout); stop(); message(`Recognition unavailable: ${data.message}. Retry the camera.`, true); }
    };
    worker.onerror = event => { event.preventDefault(); clearTimeout(timeout); if (token === generation) { stop(); message('Recognition could not load. Check your connection, then retry.', true); } };
    worker.postMessage({type: 'init'});
    animation = requestAnimationFrame(tick);
    await cameras();
  } catch (error) { if (token === generation) { stop(); message(cameraError(error), true); } }
}

function loadVoices() {
  if (!('speechSynthesis' in window)) { $('speech-message').textContent = 'Speech playback is unavailable in this browser.'; renderSentence(); return; }
  const selected = $('voice').value;
  voices = speechSynthesis.getVoices(); $('voice').replaceChildren(new Option('Browser default voice', ''));
  voices.forEach((v, i) => $('voice').add(new Option(`${v.name} (${v.lang})`, String(i))));
  if (selected && Number(selected) < voices.length) $('voice').value = selected;
}
function speak() {
  if (!sentence.trim() || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(sentence);
  if ($('voice').value !== '') utterance.voice = voices[Number($('voice').value)];
  speaking = true; renderSentence(); $('speech-message').textContent = 'Speaking…';
  utterance.onend = () => { speaking = false; renderSentence(); $('speech-message').textContent = 'Finished speaking.'; };
  utterance.onerror = event => { speaking = false; renderSentence(); $('speech-message').textContent = `Speech could not finish (${event.error}). Try another voice.`; };
  speechSynthesis.speak(utterance);
}
$('start').addEventListener('click', start); $('stop').addEventListener('click', () => stop());
$('capture').addEventListener('click', capture);
$('space').addEventListener('click', () => { if (sentence && !sentence.endsWith(' ')) sentence += ' '; renderSentence(); });
$('delete').addEventListener('click', () => { sentence = sentence.slice(0, -1); renderSentence(); });
$('clear').addEventListener('click', () => { sentence = ''; window.speechSynthesis?.cancel(); speaking = false; $('speech-message').textContent = ''; renderSentence(); });
$('speak').addEventListener('click', speak);
document.addEventListener('keydown', event => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT|BUTTON/.test(event.target.tagName)) return;
  const actions = {Enter: 'capture', ' ': 'space', Backspace: 'delete'};
  if (actions[event.key]) { event.preventDefault(); $(actions[event.key]).click(); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden && (running || stream || $('start').disabled)) stop('Camera paused while this tab was hidden. Start it again to continue.'); });
window.addEventListener('pagehide', () => { stop(); window.speechSynthesis?.cancel(); });
if ('speechSynthesis' in window) speechSynthesis.addEventListener('voiceschanged', loadVoices);
navigator.mediaDevices?.addEventListener('devicechange', () => cameras().catch(() => {}));
loadVoices(); renderSentence(); renderRecognition(); cameras().catch(() => {});

// Optional agent access uses the same capture state as the visible controls.
if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  for (const tool of [
    {name: 'read_recognition', description: 'Read the current stable letter and captured sentence.', annotations: {readOnlyHint: true}, execute: () => ({recognition: gate.snapshot(), sentence, cameraOn: running})},
    {name: 'capture_ready_letter', description: 'Capture the currently ready letter into the sentence. Does not start the camera.', annotations: {readOnlyHint: false}, execute: capture}
  ]) {
    try { Promise.resolve(document.modelContext.registerTool({...tool, inputSchema: {type: 'object', properties: {}, additionalProperties: false}, execute: input => { if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).length) throw new Error('Expected an empty object'); return tool.execute(); }}, {signal: lifecycle.signal})).catch(() => {}); } catch { /* Optional browser capability. */ }
  }
  window.addEventListener('pagehide', () => lifecycle.abort(), {once: true});
}
