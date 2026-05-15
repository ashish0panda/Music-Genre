import React, { useState, useRef, useCallback } from 'react';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import WaveformVisualizer from './components/WaveformVisualizer';
import GenreResult from './components/GenreResult';

const API_BASE = '/';

// Record icon
const MicIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    <line x1="12" y1="19" x2="12" y2="23"/>
    <line x1="8" y1="23" x2="16" y2="23"/>
  </svg>
);

const StopIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
    <rect x="4" y="4" width="16" height="16" rx="2"/>
  </svg>
);

const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);

const AnalyzeIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/>
    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);

function RecordVinyl({ spinning }) {
  return (
    <div style={{ position: 'relative', width: '120px', height: '120px', margin: '0 auto' }}>
      {/* Outer glow rings */}
      {spinning && (
        <>
          <div style={{
            position: 'absolute', inset: '-8px', borderRadius: '50%',
            border: '1px solid rgba(232,255,71,0.3)',
            animation: 'pulse-ring 2s ease-out infinite',
          }} />
          <div style={{
            position: 'absolute', inset: '-4px', borderRadius: '50%',
            border: '1px solid rgba(232,255,71,0.2)',
            animation: 'pulse-ring 2s ease-out 0.5s infinite',
          }} />
        </>
      )}
      {/* Vinyl record */}
      <svg
        viewBox="0 0 120 120"
        style={{
          width: '100%', height: '100%',
          animation: spinning ? 'record-spin 3s linear infinite' : 'none',
          filter: spinning ? 'drop-shadow(0 0 16px rgba(232,255,71,0.4))' : 'none',
          transition: 'filter 0.3s',
        }}
      >
        {/* Outer record */}
        <circle cx="60" cy="60" r="58" fill="#1a1a2e" stroke="#2a2a40" strokeWidth="1"/>
        {/* Grooves */}
        {[46, 40, 34, 28, 22].map((r, i) => (
          <circle key={i} cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="2"/>
        ))}
        {/* Label */}
        <circle cx="60" cy="60" r="18" fill="#e8ff47"/>
        <circle cx="60" cy="60" r="4" fill="#080810"/>
        <text x="60" y="56" textAnchor="middle" fill="#080810" fontSize="5" fontWeight="700" fontFamily="'DM Sans', sans-serif">GENRE</text>
        <text x="60" y="63" textAnchor="middle" fill="#080810" fontSize="5" fontWeight="700" fontFamily="'DM Sans', sans-serif">SCOPE</text>
      </svg>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '24px', padding: '40px 0' }}>
      <div style={{ position: 'relative', width: '72px', height: '72px' }}>
        <div style={{
          position: 'absolute', inset: 0, borderRadius: '50%',
          border: '2px solid rgba(255,255,255,0.05)',
        }} />
        <div style={{
          position: 'absolute', inset: 0, borderRadius: '50%',
          border: '2px solid transparent',
          borderTopColor: 'var(--accent)',
          animation: 'spin-slow 1s linear infinite',
        }} />
        <div style={{
          position: 'absolute', inset: '8px', borderRadius: '50%',
          border: '2px solid transparent',
          borderBottomColor: 'var(--accent2)',
          animation: 'spin-reverse 0.8s linear infinite',
        }} />
        <div style={{
          position: 'absolute', inset: '50%', translate: '-50% -50%',
          width: '8px', height: '8px', borderRadius: '50%',
          background: 'var(--accent)',
          boxShadow: '0 0 12px var(--accent)',
        }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '16px', color: 'var(--text)', marginBottom: '4px' }}>Analyzing audio...</div>
        <div style={{ fontSize: '13px', color: 'var(--text-dim)' }}>Running ML classification model</div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('record'); // 'record' | 'upload'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [resultFilename, setResultFilename] = useState('');
  const fileInputRef = useRef(null);

  const { isRecording, recordingTime, audioBlob, error: recError, startRecording, stopRecording, reset: resetRecorder } = useAudioRecorder();

  const resetAll = () => {
    setResult(null);
    setError(null);
    setUploadedFile(null);
    setResultFilename('');
    resetRecorder();
  };

  const classifyAudio = useCallback(async (blob, filename) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setResultFilename(filename || 'recording');

    try {
      const formData = new FormData();
      const contentType = blob.type || 'audio/webm';
      const ext = contentType.includes('webm') ? 'webm' : contentType.includes('mp3') ? 'mp3' : 'wav';
      formData.append('file', blob, filename || `recording.${ext}`);

      const res = await fetch('/classify', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to classify audio. Make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleFileUpload = (file) => {
    if (!file) return;
    if (!file.type.startsWith('audio/')) {
      setError('Please upload a valid audio file (mp3, wav, ogg, flac, m4a).');
      return;
    }
    setUploadedFile(file);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const formatTime = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  const canAnalyzeRecord = audioBlob && !isLoading;
  const canAnalyzeUpload = uploadedFile && !isLoading;

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg)',
    }}>
      {/* Background gradient blobs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: '-20%', left: '-10%',
          width: '600px', height: '600px', borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(232,255,71,0.04) 0%, transparent 70%)',
        }} />
        <div style={{
          position: 'absolute', bottom: '-20%', right: '-10%',
          width: '500px', height: '500px', borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,79,216,0.04) 0%, transparent 70%)',
        }} />
        <div style={{
          position: 'absolute', top: '40%', right: '20%',
          width: '300px', height: '300px', borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(79,255,176,0.03) 0%, transparent 70%)',
        }} />
      </div>

      <div style={{
        position: 'relative', zIndex: 1,
        maxWidth: '680px', margin: '0 auto', width: '100%',
        padding: '40px 20px 80px',
        display: 'flex', flexDirection: 'column', gap: '32px',
      }}>

        {/* Header */}
        <header style={{ textAlign: 'center', animation: 'fadeUp 0.6s ease both' }}>
          <RecordVinyl spinning={isRecording || isLoading} />
          <div style={{ marginTop: '20px' }}>
            <h1 style={{
              fontFamily: 'var(--font-display)',
              fontSize: 'clamp(48px, 10vw, 80px)',
              letterSpacing: '0.05em',
              color: 'var(--text)',
              lineHeight: 1,
            }}>
              GENRE<span style={{ color: 'var(--accent)' }}>SCOPE</span>
            </h1>
            <p style={{ fontSize: '14px', color: 'var(--text-dim)', marginTop: '8px', fontWeight: 300 }}>
              AI-powered music genre classification · 30-second analysis
            </p>
          </div>
        </header>

        {/* Main Card */}
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          overflow: 'hidden',
          animation: 'fadeUp 0.6s ease 0.1s both',
        }}>
          {/* Tabs */}
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr',
            borderBottom: '1px solid var(--border)',
          }}>
            {['record', 'upload'].map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); resetAll(); }}
                style={{
                  background: tab === t ? 'rgba(232,255,71,0.06)' : 'transparent',
                  border: 'none',
                  borderBottom: tab === t ? '2px solid var(--accent)' : '2px solid transparent',
                  color: tab === t ? 'var(--accent)' : 'var(--text-dim)',
                  padding: '16px',
                  fontSize: '13px',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  cursor: 'pointer',
                  fontFamily: 'var(--font-body)',
                  transition: 'all 0.2s',
                }}
              >
                {t === 'record' ? '🎙 Record' : '📁 Upload'}
              </button>
            ))}
          </div>

          <div style={{ padding: '32px' }}>
            {/* RECORD TAB */}
            {tab === 'record' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', alignItems: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '14px', color: 'var(--text-dim)', marginBottom: '4px' }}>
                    Record up to 30 seconds of music for instant genre detection
                  </p>
                </div>

                {/* Timer */}
                <div style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '52px',
                  color: isRecording ? 'var(--accent)' : 'var(--text-dimmer)',
                  letterSpacing: '0.05em',
                  lineHeight: 1,
                  transition: 'color 0.3s',
                  textShadow: isRecording ? '0 0 30px rgba(232,255,71,0.4)' : 'none',
                }}>
                  {formatTime(recordingTime)} <span style={{ fontSize: '20px' }}>/ 0:30</span>
                </div>

                {/* Waveform */}
                <WaveformVisualizer isActive={isRecording} />

                {/* Record button */}
                {!audioBlob ? (
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    style={{
                      width: '72px', height: '72px', borderRadius: '50%',
                      border: '2px solid',
                      borderColor: isRecording ? '#ff4747' : 'var(--accent)',
                      background: isRecording
                        ? 'rgba(255,71,71,0.15)'
                        : 'rgba(232,255,71,0.1)',
                      color: isRecording ? '#ff4747' : 'var(--accent)',
                      cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'all 0.2s',
                      animation: isRecording ? 'glow-pulse 2s ease infinite' : 'none',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.05)'; }}
                    onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; }}
                  >
                    {isRecording ? <StopIcon /> : <MicIcon />}
                  </button>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', width: '100%' }}>
                    <div style={{ fontSize: '13px', color: 'var(--accent3)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span>✓</span> Recording captured ({recordingTime}s)
                    </div>
                    <audio
                      src={URL.createObjectURL(audioBlob)}
                      controls
                      style={{ width: '100%', borderRadius: 'var(--radius-sm)', height: '40px', accentColor: 'var(--accent)' }}
                    />
                    <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
                      <button
                        onClick={resetAll}
                        style={{
                          flex: 1, padding: '12px', borderRadius: 'var(--radius-sm)',
                          background: 'transparent', border: '1px solid var(--border)',
                          color: 'var(--text-dim)', cursor: 'pointer', fontSize: '13px',
                          fontFamily: 'var(--font-body)', transition: 'border-color 0.2s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'}
                        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                      >
                        Re-record
                      </button>
                      <button
                        onClick={() => classifyAudio(audioBlob, `recording_${recordingTime}s.webm`)}
                        disabled={isLoading}
                        style={{
                          flex: 2, padding: '12px', borderRadius: 'var(--radius-sm)',
                          background: 'var(--accent)', border: 'none',
                          color: '#080810', cursor: 'pointer', fontSize: '14px',
                          fontWeight: 600, fontFamily: 'var(--font-body)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                          transition: 'opacity 0.2s, transform 0.1s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
                        onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
                      >
                        <AnalyzeIcon /> Analyze Genre
                      </button>
                    </div>
                  </div>
                )}

                {recError && (
                  <div style={{ fontSize: '13px', color: '#ff6b6b', textAlign: 'center', padding: '8px', background: 'rgba(255,107,107,0.08)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,107,107,0.2)', width: '100%' }}>
                    {recError}
                  </div>
                )}
              </div>
            )}

            {/* UPLOAD TAB */}
            {tab === 'upload' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: `2px dashed ${isDragging ? 'var(--accent)' : uploadedFile ? 'var(--accent3)' : 'rgba(255,255,255,0.12)'}`,
                    borderRadius: 'var(--radius)',
                    padding: '48px 24px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: isDragging ? 'rgba(232,255,71,0.04)' : uploadedFile ? 'rgba(79,255,176,0.03)' : 'rgba(255,255,255,0.01)',
                    transition: 'all 0.2s',
                    animation: isDragging ? 'glow-pulse 1s ease infinite' : 'none',
                  }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileUpload(e.target.files[0])}
                  />
                  {uploadedFile ? (
                    <>
                      <div style={{ fontSize: '36px', marginBottom: '8px' }}>🎵</div>
                      <div style={{ fontSize: '14px', color: 'var(--accent3)', fontWeight: 500, marginBottom: '4px' }}>
                        {uploadedFile.name}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB · Click to change
                      </div>
                    </>
                  ) : (
                    <>
                      <div style={{ color: 'var(--text-dim)', marginBottom: '12px' }}>
                        <UploadIcon />
                      </div>
                      <div style={{ fontSize: '15px', color: 'var(--text)', marginBottom: '6px' }}>
                        Drop audio file here
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                        MP3, WAV, FLAC, OGG, M4A · Max 50MB
                      </div>
                    </>
                  )}
                </div>

                {uploadedFile && (
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      onClick={resetAll}
                      style={{
                        flex: 1, padding: '12px', borderRadius: 'var(--radius-sm)',
                        background: 'transparent', border: '1px solid var(--border)',
                        color: 'var(--text-dim)', cursor: 'pointer', fontSize: '13px',
                        fontFamily: 'var(--font-body)',
                      }}
                    >
                      Clear
                    </button>
                    <button
                      onClick={() => classifyAudio(uploadedFile, uploadedFile.name)}
                      disabled={isLoading}
                      style={{
                        flex: 2, padding: '12px', borderRadius: 'var(--radius-sm)',
                        background: 'var(--accent)', border: 'none',
                        color: '#080810', cursor: 'pointer', fontSize: '14px',
                        fontWeight: 600, fontFamily: 'var(--font-body)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                        opacity: isLoading ? 0.6 : 1,
                        transition: 'opacity 0.2s, transform 0.1s',
                      }}
                      onMouseEnter={e => !isLoading && (e.currentTarget.style.transform = 'translateY(-1px)')}
                      onMouseLeave={e => (e.currentTarget.style.transform = 'translateY(0)')}
                    >
                      <AnalyzeIcon /> Analyze Genre
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Loading */}
        {isLoading && (
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            animation: 'fadeUp 0.4s ease both',
          }}>
            <LoadingSpinner />
          </div>
        )}

        {/* Error */}
        {error && !isLoading && (
          <div style={{
            background: 'rgba(255,71,71,0.06)',
            border: '1px solid rgba(255,71,71,0.2)',
            borderRadius: 'var(--radius)',
            padding: '20px 24px',
            fontSize: '14px',
            color: '#ff8a8a',
            animation: 'fadeUp 0.4s ease both',
          }}>
            <strong>Error:</strong> {error}
            {error.includes('backend') && (
              <div style={{ marginTop: '8px', fontSize: '12px', color: '#ff6b6b80' }}>
                Tip: Make sure the backend is running with <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 6px', borderRadius: '4px' }}>uvicorn main:app --reload</code>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {result && !isLoading && (
          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '32px',
            animation: 'fadeUp 0.5s ease both',
          }}>
            <GenreResult result={result} filename={resultFilename} />
            <button
              onClick={resetAll}
              style={{
                marginTop: '24px',
                width: '100%',
                padding: '12px',
                borderRadius: 'var(--radius-sm)',
                background: 'transparent',
                border: '1px solid var(--border)',
                color: 'var(--text-dim)',
                cursor: 'pointer',
                fontSize: '13px',
                fontFamily: 'var(--font-body)',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'; e.currentTarget.style.color = 'var(--text)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-dim)'; }}
            >
              ← Analyze another track
            </button>
          </div>
        )}

        {/* Footer info */}
        <div style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-dimmer)', animation: 'fadeUp 0.6s ease 0.2s both' }}>
          Powered by Hugging Face Transformers · Librosa · FastAPI
        </div>
      </div>
    </div>
  );
}
