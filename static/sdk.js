/**
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║        SMART TEACHER — SDK JavaScript Client                       ║
 * ║                                                                      ║
 * ║  Usage minimal :                                                     ║
 * ║    const st = new SmartTeacher({ host: 'localhost:8000' })          ║
 * ║    await st.connect()                                                ║
 * ║    st.onTranscription = text => console.log(text)                   ║
 * ║    st.onAnswer        = text => console.log(text)                   ║
 * ║    await st.startMic()                                               ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 */

class SmartTeacher {
  constructor(options = {}) {
    this.host       = options.host       || location.host;
    this.language   = options.language   || 'fr';
    this.level      = options.level      || 'lycée';
    this.sessionId  = options.sessionId  || crypto.randomUUID();

    // Callbacks publics
    this.onConnected       = options.onConnected       || (() => {});
    this.onDisconnected    = options.onDisconnected    || (() => {});
    this.onTranscription   = options.onTranscription   || (() => {});
    this.onAnswer          = options.onAnswer          || (() => {});
    this.onAudio           = options.onAudio           || (() => {});
    this.onStateChange     = options.onStateChange     || (() => {});
    this.onSlideUpdate     = options.onSlideUpdate     || (() => {});
    this.onPerformance     = options.onPerformance     || (() => {});
    this.onSystemNotice    = options.onSystemNotice    || (() => {});
    this.onPresentationPlan = options.onPresentationPlan || (() => {});  // ✅ Plan agentique
    this.onQaIntent        = options.onQaIntent        || (() => {});    // ✅ Q&A intent classifié
    this.onError           = options.onError           || ((e) => console.error(e));

    // Interne
    this._ws           = null;
    this._recorder     = null;
    this._audioBuffer  = [];
    this._currentAudio = null;
    this._recording    = false;
    this._sessionStarted = false;
  }

  // ── Connexion WebSocket ─────────────────────────────────────────────
  async connect() {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      this._ws = new WebSocket(`${proto}://${this.host}/ws/${this.sessionId}`);

      this._ws.onopen = () => {
        this._startSession();
        this.onConnected(this.sessionId);
        resolve(this);
      };

      this._ws.onclose = () => {
        this.onDisconnected();
        setTimeout(() => this.connect(), 3000);
      };

      this._ws.onerror = (e) => {
        this.onError(e);
        reject(e);
      };

      this._ws.onmessage = (ev) => {
        try { this._handleMessage(JSON.parse(ev.data)); }
        catch(e) { this.onError(e); }
      };
    });
  }

  disconnect() {
    if (this._ws) { this._ws.close(); this._ws = null; }
  }

  // ── Messages WebSocket ──────────────────────────────────────────────
  _send(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN)
      this._ws.send(JSON.stringify(obj));
  }

  _startSession() {
    if (!this._sessionStarted) {
      this._send({ type: 'start_session', language: this.language, level: this.level });
      this._sessionStarted = true;
    }
  }

  _handleMessage(msg) {
    switch (msg.type) {
      case 'state_change':
        // ✅ PASS FULL MESSAGE OBJECT SO UI CAN DISPLAY METRICS
        this.onStateChange(msg);
        break;
      case 'transcription':
        this.onTranscription(msg.text, msg.lang, msg.confidence);
        break;
      case 'answer_text':
        this._pendingAnswer = msg.text;
        this.onAnswer(msg.text, msg.subject);
        break;
      case 'audio_chunk':
        this._bufferAudio(msg.data, msg.mime, msg.final);
        break;
      case 'slide_update':
        this.onSlideUpdate(msg);
        break;
      case 'presentation_plan':
        this.onPresentationPlan(msg);
        break;
      case 'qa_intent':
        this.onQaIntent(msg);
        break;
      case 'performance':
        this.onPerformance(msg);
        break;
      case 'system_notice':
        this.onSystemNotice(msg.text, msg);
        break;
      case 'error':
        this.onError(msg.message);
        break;
    }
  }

  // ── Microphone ──────────────────────────────────────────────────────
  async startMic() {
    if (this._recording) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });
      
      // ✅ DEBUG: Check if we got audio tracks
      console.log('🎤 Audio stream tracks:', stream.getAudioTracks().length);
      const audioTrack = stream.getAudioTracks()[0];
      console.log('🎤 Audio track settings:', audioTrack.getSettings());
      
      const mime   = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                     ? 'audio/webm;codecs=opus' : 'audio/webm';
      
      console.log('🎤 Using MIME type:', mime);

      this._recorder = new MediaRecorder(stream, { mimeType: mime });

      this._recorder.ondataavailable = (e) => {
        console.log('🎤 Audio chunk received, size:', e.data.size);
        if (e.data.size > 0) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const b64 = reader.result.split(',')[1];
            console.log('🎤 Sending audio chunk, b64 length:', b64.length);
            this._send({ type: 'audio_chunk', data: b64 });
          };
          reader.readAsDataURL(e.data);
        } else {
          console.warn('🎤 Empty audio chunk received');
        }
      };

      this._recorder.onstop = () => {
        console.log('🎤 Recording stopped');
        stream.getTracks().forEach(t => t.stop());
        this._send({ type: 'audio_end' });
        this._recording = false;
      };

      this._recorder.onerror = (e) => {
        console.error('🎤 MediaRecorder error:', e);
        this.onError('MediaRecorder error: ' + e.error?.message);
      };

      this._recorder.start(300);
      this._recording = true;
      this._send({ type: 'interrupt' }); // interrompt la présentation
      
      console.log('🎤 Recording started successfully');
      return true;
      
    } catch (error) {
      console.error('🎤 Failed to start microphone:', error);
      this.onError('Microphone access failed: ' + error.message);
      throw error;
    }
  }

  stopMic() {
    if (this._recorder && this._recorder.state !== 'inactive') {
      this._recorder.stop();
    }
  }

  isRecording() { return this._recording; }

  // ── Texte ───────────────────────────────────────────────────────────
  sendText(text) {
    this._startSession();
    this._send({ type: 'text', content: text });
  }

  // ── Navigation cours ────────────────────────────────────────────────
  nextSection()  { this._send({ type: 'next_section' }); }
  prevSection()  { this._send({ type: 'prev_section' }); }
  interrupt()    {
    // Compute the REAL audio playback position so the backend can save
    // an accurate resume cursor. Without this the backend saves
    // cursor = len(narration_text) once text streaming finishes, even
    // if the audio is still playing — and the resume restarts the
    // whole slide. With audio_progress in [0, 1], the backend converts
    // it to a char offset that matches what the student actually heard.
    let progress = null;
    const a = this._currentAudio;
    // ⛳ Diagnostic : show what the SDK sees about the audio element
    // when interrupt() is called. If you don't see this in the browser
    // console, the page is serving an old sdk.js (force reload with
    // Ctrl+Shift+R or open in private tab).
    console.log('⛳ SDK interrupt() called', {
      hasAudio: !!a,
      currentTime: a ? a.currentTime : null,
      duration: a ? a.duration : null,
      paused: a ? a.paused : null,
      ended: a ? a.ended : null,
      readyState: a ? a.readyState : null,
    });
    if (a && isFinite(a.duration) && a.duration > 0) {
      progress = Math.min(1, Math.max(0, a.currentTime / a.duration));
    }
    const msg = { type: 'interrupt' };
    if (progress !== null) msg.audio_progress = progress;
    console.log('⛳ SDK interrupt() sending', msg);
    this._send(msg);
    this.stopAudio();
  }
  ping()         { this._send({ type: 'ping' }); }

  // ── Audio playback ──────────────────────────────────────────────────
  _bufferAudio(b64, mime, final) {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    this._audioBuffer.push(bytes);

    if (final) {
      const total = this._audioBuffer.reduce((s, c) => s + c.length, 0);
      const merged = new Uint8Array(total);
      let offset = 0;
      this._audioBuffer.forEach(c => { merged.set(c, offset); offset += c.length; });
      this._audioBuffer = [];

      const blob = new Blob([merged], { type: mime || 'audio/mpeg' });
      const url  = URL.createObjectURL(blob);
      this.playAudio(url);
      this.onAudio(url);
    }
  }

  playAudio(url) {
    this.stopAudio();
    this._currentAudio = new Audio(url);
    this._currentAudio.play().catch(e => this.onError('Audio play failed: ' + e));
  }

  stopAudio() {
    if (this._currentAudio) { this._currentAudio.pause(); this._currentAudio = null; }
    this._audioBuffer = [];
  }

  // ── Test microphone ─────────────────────────────────────────────────
  async testMic() {
    try {
      console.log('🧪 Testing microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });
      
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      microphone.connect(analyser);
      
      analyser.fftSize = 256;
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      // Check for audio levels
      let hasAudio = false;
      for (let i = 0; i < 10; i++) {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b) / bufferLength;
        console.log(`🧪 Audio level ${i}: ${avg.toFixed(2)}`);
        if (avg > 5) { // Some threshold for audio detection
          hasAudio = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      stream.getTracks().forEach(track => track.stop());
      audioContext.close();
      
      if (hasAudio) {
        console.log('✅ Microphone test passed - audio detected');
        return true;
      } else {
        console.warn('⚠️ Microphone test failed - no audio detected (speak into mic)');
        return false;
      }
      
    } catch (error) {
      console.error('❌ Microphone test failed:', error);
      return false;
    }
  }
  async askText(question) {
    const fd = new FormData();
    fd.append('question', question);
    const r = await fetch(`//${this.host}/ask`, {
      method: 'POST',
      headers: { 'X-Session-ID': this.sessionId },
      body: fd,
    });
    return r.json();
  }

  async uploadCourse(file, language = 'fr', level = 'lycée') {
    const fd = new FormData();
    fd.append('files', file);
    fd.append('language', language);
    fd.append('level', level);
    const r = await fetch(`//${this.host}/course/build`, { method: 'POST', body: fd });
    return r.json();
  }

  async listCourses() {
    const r = await fetch(`//${this.host}/course/list`);
    return r.json();
  }

  async getCourseStructure(courseId) {
    const r = await fetch(`//${this.host}/course/${courseId}/structure`);
    return r.json();
  }

  async getHealth() {
    const r = await fetch(`//${this.host}/health`);
    return r.json();
  }
}

// Export pour modules ES6 et CommonJS
if (typeof module !== 'undefined' && module.exports) module.exports = SmartTeacher;
if (typeof window !== 'undefined') window.SmartTeacher = SmartTeacher;
