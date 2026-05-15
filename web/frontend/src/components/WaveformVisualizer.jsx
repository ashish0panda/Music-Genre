import React, { useEffect, useRef } from 'react';

export default function WaveformVisualizer({ isActive, color = '#e8ff47' }) {
  const bars = 32;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '3px',
      height: '48px',
    }}>
      {Array.from({ length: bars }).map((_, i) => (
        <div
          key={i}
          style={{
            width: '3px',
            borderRadius: '99px',
            backgroundColor: color,
            height: isActive ? undefined : '4px',
            animation: isActive
              ? `waveform ${0.4 + Math.random() * 0.6}s ease-in-out ${i * 0.03}s infinite alternate`
              : 'none',
            minHeight: '4px',
            maxHeight: '44px',
            opacity: isActive ? 0.85 : 0.25,
            transition: 'opacity 0.3s',
          }}
        />
      ))}
    </div>
  );
}
