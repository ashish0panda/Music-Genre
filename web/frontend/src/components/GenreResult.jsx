import React from 'react';

const GENRE_COLORS = {
  default: '#e8ff47', rock: '#ff6b35', pop: '#ff4fd8',
  jazz: '#4fffb0', classical: '#b4a0ff', electronic: '#47e8ff',
  hip: '#ffb347', metal: '#ff4747', country: '#ffd700',
  blues: '#4f9fff', reggae: '#4fff8f', folk: '#d4a76a',
  soul: '#ff7eb3', funk: '#ff9f43', latin: '#ff6b6b', indie: '#c084fc',
  ambient: '#a0d4ff', soundtrack: '#ffc0a0', 'r&b': '#ff9fd4',
};

const MOOD_EMOJI = {
  "Romantic": "💕", "Happy": "😊", "Sad": "😢", "Energetic": "⚡",
  "Calm": "🌊", "Melancholic": "🌧️", "Aggressive": "🔥", "Upbeat": "🎉",
  "Dark": "🌑", "Peaceful": "🕊️", "Uplifting": "✨", "Smooth": "🎶", "Bright": "☀️",
};

function getGenreColor(genre) {
  if (!genre) return GENRE_COLORS.default;
  const lower = genre.toLowerCase();
  for (const [key, color] of Object.entries(GENRE_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return GENRE_COLORS.default;
}

function MoodTag({ mood, confidence }) {
  const emoji = MOOD_EMOJI[mood] || '🎵';
  const strength = confidence / 100;
  return (
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: '5px',
        background: `rgba(79,255,176,${strength * 0.12})`,
        border: `1px solid rgba(79,255,176,${strength * 0.5})`,
        borderRadius: '99px',
        padding: '6px 14px',
        fontSize: '13px',
        color: `rgba(160,255,220,${0.5 + strength * 0.5})`,
        fontWeight: confidence > 80 ? 500 : 400,
        transition: 'all 0.2s',
      }}>
        <span style={{ fontSize: '15px' }}>{emoji}</span>
        {mood}
        <span style={{ fontSize: '11px', opacity: 0.6, marginLeft: '2px' }}>{confidence}%</span>
      </div>
  );
}

function FeaturePill({ label, value, unit = '' }) {
  return (
      <div style={{
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '10px',
        padding: '12px 16px',
        display: 'flex', flexDirection: 'column', gap: '4px',
      }}>
      <span style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 500 }}>
        {label}
      </span>
        <span style={{ fontSize: '18px', fontWeight: 500, color: 'var(--text)' }}>
        {value}<span style={{ fontSize: '12px', color: 'var(--text-dim)', marginLeft: '2px' }}>{unit}</span>
      </span>
      </div>
  );
}

export default function GenreResult({ result, filename }) {
  if (!result) return null;

  const { predictions = [], top_genre, confidence, features = {}, moods = [], models_used = [], inference_time_sec } = result;
  const accentColor = getGenreColor(top_genre);

  return (
      <div style={{ animation: 'fadeUp 0.5s ease forwards' }}>
        {/* Top Genre Hero */}
        <div style={{
          background: `radial-gradient(ellipse at 30% 0%, ${accentColor}18 0%, transparent 60%), var(--surface)`,
          border: `1px solid ${accentColor}30`,
          borderRadius: 'var(--radius)',
          padding: '32px',
          marginBottom: '16px',
          position: 'relative',
          overflow: 'hidden',
        }}>
          <div style={{ position: 'absolute', top: '-40px', right: '-40px', width: '180px', height: '180px', borderRadius: '50%', border: `2px solid ${accentColor}20`, pointerEvents: 'none' }} />
          <div style={{ position: 'absolute', top: '-20px', right: '-20px', width: '140px', height: '140px', borderRadius: '50%', border: `1px solid ${accentColor}15`, pointerEvents: 'none' }} />

          <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: '8px' }}>
            Top Genre
          </div>
          <div style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(42px, 8vw, 72px)',
            lineHeight: 1, color: accentColor,
            letterSpacing: '0.02em', marginBottom: '8px',
            textShadow: `0 0 40px ${accentColor}40`,
          }}>
            {top_genre}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <div style={{ fontSize: '22px', fontWeight: 500, color: 'var(--text)' }}>{confidence}%</div>
            <div style={{ fontSize: '13px', color: 'var(--text-dim)' }}>confidence</div>
            {filename && (
                <>
                  <div style={{ color: 'var(--text-dimmer)' }}>·</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-dim)', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{filename}</div>
                </>
            )}
            {models_used.length > 0 && (
                <>
                  <div style={{ color: 'var(--text-dimmer)' }}>·</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-dimmer)' }}>
                    {models_used.length} model{models_used.length > 1 ? 's' : ''} ensemble
                  </div>
                </>
            )}
          </div>
        </div>

        {/* Mood / Vibe Section */}
        {moods.length > 0 && (
            <div style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              padding: '20px 24px',
              marginBottom: '16px',
              animation: 'fadeUp 0.5s ease 0.1s both',
            }}>
              <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '14px' }}>
                🎭 Mood & Vibe
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {moods.map((m, i) => (
                    <MoodTag key={i} mood={m.mood} confidence={m.confidence} />
                ))}
              </div>
              <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--text-dimmer)' }}>
                Detected from tempo, key, energy & spectral features
              </div>
            </div>
        )}

        {/* Audio Features */}
        {Object.keys(features).length > 0 && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
              gap: '10px', marginBottom: '16px',
              animation: 'fadeUp 0.5s ease 0.15s both',
            }}>
              {features.tempo_bpm && <FeaturePill label="Tempo" value={features.tempo_bpm} unit="BPM" />}
              {features.key && <FeaturePill label="Key" value={features.key} />}
              {features.duration_sec && <FeaturePill label="Duration" value={features.duration_sec} unit="s" />}
              {features.energy !== undefined && <FeaturePill label="Energy" value={features.energy} unit="%" />}
            </div>
        )}

        {/* All Genre Predictions */}
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '24px', marginBottom: '12px',
          animation: 'fadeUp 0.5s ease 0.2s both',
        }}>
          <div style={{ fontSize: '11px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '20px' }}>
            All Genre Predictions
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {predictions.map((p, i) => {
              const color = i === 0 ? accentColor : getGenreColor(p.genre);
              return (
                  <div key={i} style={{ animation: `slideIn 0.4s ease ${i * 0.07}s both` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '14px', color: i === 0 ? 'var(--text)' : 'var(--text-dim)', fontWeight: i === 0 ? 500 : 400 }}>
                    {p.genre}
                  </span>
                      <span style={{ fontSize: '13px', color: i === 0 ? color : 'var(--text-dim)', fontWeight: 500 }}>
                    {p.confidence}%
                  </span>
                    </div>
                    <div style={{ height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '99px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', width: `${p.confidence}%`,
                        background: i === 0 ? `linear-gradient(90deg, ${color}80, ${color})` : 'rgba(255,255,255,0.2)',
                        borderRadius: '99px',
                        animation: `fillBar 0.8s cubic-bezier(0.16,1,0.3,1) ${0.3 + i * 0.07}s both`,
                        '--target-width': `${p.confidence}%`,
                      }} />
                    </div>
                  </div>
              );
            })}
          </div>
        </div>

        <div style={{ fontSize: '11px', color: 'var(--text-dimmer)', textAlign: 'right' }}>
          Analyzed in {inference_time_sec}s · {models_used.join(' + ')} ensemble
        </div>
      </div>
  );
}