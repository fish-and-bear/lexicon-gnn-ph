import React from 'react';
import './NetworkControls.css';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (newDepth: number) => void;
  onBreadthChange: (newBreadth: number) => void;
}

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth = 1,
  breadth = 15,
  onDepthChange,
  onBreadthChange
}) => {
  const safeDepth = Math.max(1, Math.min(5, depth));
  const safeBreadth = Math.max(5, Math.min(75, breadth));

  return (
    <div className="network-controls">
      <h3>Network Settings</h3>
      
      <div className="slider-container">
        <label htmlFor="depth-slider">Relation Depth: {safeDepth}</label>
        <input
          id="depth-slider"
          type="range"
          min={1}
          max={5}
          step={1}
          value={safeDepth}
          onChange={(e) => {
            const newValue = parseInt(e.target.value);
            if (onDepthChange && !isNaN(newValue)) {
              onDepthChange(newValue);
            }
          }}
        />
        {safeDepth > 3 && (
          <div className="high-value-warning">
            Deep searches may take longer to load and display.
          </div>
        )}
      </div>
      
      <div className="slider-container">
        <label htmlFor="breadth-slider">Display Breadth: {safeBreadth}</label>
        <input
          id="breadth-slider"
          type="range"
          min={5}
          max={75}
          step={5}
          value={safeBreadth}
          onChange={(e) => {
            const newValue = parseInt(e.target.value);
            if (onBreadthChange && !isNaN(newValue)) {
              onBreadthChange(newValue);
            }
          }}
        />
        {safeBreadth > 40 && (
          <div className="high-value-warning">
            High breadth values create dense, complex networks.
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkControls; 