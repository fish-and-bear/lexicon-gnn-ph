import React from 'react';
import './NetworkControls.css';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (newDepth: number) => void;
  onBreadthChange: (newBreadth: number) => void;
}

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange
}) => {
  return (
    <div className="network-controls">
      <h3>Network Settings</h3>
      
      <div className="slider-container">
        <label htmlFor="depth-slider">Relation Depth: {depth}</label>
        <input
          id="depth-slider"
          type="range"
          min={1}
          max={5}
          step={1}
          value={depth}
          onChange={(e) => onDepthChange(parseInt(e.target.value))}
        />
        {depth > 3 && (
          <div className="high-value-warning">
            Deep searches may take longer to load and display.
          </div>
        )}
      </div>
      
      <div className="slider-container">
        <label htmlFor="breadth-slider">Display Breadth: {breadth}</label>
        <input
          id="breadth-slider"
          type="range"
          min={5}
          max={75}
          step={5}
          value={breadth}
          onChange={(e) => onBreadthChange(parseInt(e.target.value))}
        />
        {breadth > 40 && (
          <div className="high-value-warning">
            High breadth values create dense, complex networks.
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkControls; 