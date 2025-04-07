import React, { useState, useEffect, useCallback } from 'react';
import './NetworkControls.css';
import { useTheme } from "../contexts/ThemeContext";

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange
}) => {
  const { theme } = useTheme();
  const [localDepth, setLocalDepth] = useState(depth);
  const [localBreadth, setLocalBreadth] = useState(breadth);
  const [depthDescription, setDepthDescription] = useState('');
  const [breadthDescription, setBreadthDescription] = useState('');

  // Update local state when props change
  useEffect(() => {
    setLocalDepth(depth);
  }, [depth]);

  useEffect(() => {
    setLocalBreadth(breadth);
  }, [breadth]);

  // Use memo to generate descriptions to avoid recalculating on every render
  useEffect(() => {
    if (localDepth <= 1) {
      setDepthDescription('Direct connections only');
    } else if (localDepth <= 2) {
      setDepthDescription('Friends of friends');
    } else if (localDepth <= 3) {
      setDepthDescription('Extended network');
    } else if (localDepth <= 4) {
      setDepthDescription('Deep connections');
    } else {
      setDepthDescription('Very extensive network (may be slow)');
    }
  }, [localDepth]);

  useEffect(() => {
    if (localBreadth <= 5) {
      setBreadthDescription('Few connections per word');
    } else if (localBreadth <= 15) {
      setBreadthDescription('Moderate connections');
    } else if (localBreadth <= 30) {
      setBreadthDescription('Many connections');
    } else {
      setBreadthDescription('Maximum connections (may be slow)');
    }
  }, [localBreadth]);

  // Debounce the change handlers to avoid too many updates
  const debouncedDepthChange = useCallback(() => {
    if (localDepth !== depth) {
      onDepthChange(localDepth);
    }
  }, [localDepth, depth, onDepthChange]);

  const debouncedBreadthChange = useCallback(() => {
    if (localBreadth !== breadth) {
      onBreadthChange(localBreadth);
    }
  }, [localBreadth, breadth, onBreadthChange]);

  // Apply the debounce effect
  useEffect(() => {
    const timer = setTimeout(() => {
      debouncedDepthChange();
    }, 300);
    return () => clearTimeout(timer);
  }, [debouncedDepthChange]);

  useEffect(() => {
    const timer = setTimeout(() => {
      debouncedBreadthChange();
    }, 300);
    return () => clearTimeout(timer);
  }, [debouncedBreadthChange]);

  return (
    <div className={`network-controls ${theme}`}>
      <h3>Network Controls</h3>
      <div className="sliders-row">
        <div className="slider-container">
          <label htmlFor="depth-slider">Connection Depth</label>
          <div className="slider-value">{localDepth}</div>
          <input
            id="depth-slider"
            type="range"
            min="1"
            max="5"
            value={localDepth}
            onChange={(e) => setLocalDepth(parseInt(e.target.value, 10))}
          />
          <div className="slider-description">{depthDescription}</div>
        </div>
        <div className="slider-container">
          <label htmlFor="breadth-slider">Connection Breadth</label>
          <div className="slider-value">{localBreadth}</div>
          <input
            id="breadth-slider"
            type="range"
            min="1"
            max="50"
            value={localBreadth}
            onChange={(e) => setLocalBreadth(parseInt(e.target.value, 10))}
          />
          <div className="slider-description">{breadthDescription}</div>
        </div>
      </div>
    </div>
  );
};

export default NetworkControls; 