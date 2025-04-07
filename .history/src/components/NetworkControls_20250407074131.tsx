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
  const [isDraggingDepth, setIsDraggingDepth] = useState(false);
  const [isDraggingBreadth, setIsDraggingBreadth] = useState(false);

  // Update local state when props change (but not during dragging)
  useEffect(() => {
    if (!isDraggingDepth) {
      setLocalDepth(depth);
    }
  }, [depth, isDraggingDepth]);

  useEffect(() => {
    if (!isDraggingBreadth) {
      setLocalBreadth(breadth);
    }
  }, [breadth, isDraggingBreadth]);

  // Get description for current depth value
  const getDepthDescription = useCallback((value: number): string => {
    if (value <= 1) return 'Direct connections only';
    if (value <= 2) return 'Friends of friends';
    if (value <= 3) return 'Extended network';
    if (value <= 4) return 'Deep connections';
    return 'Very extensive network (may be slow)';
  }, []);

  // Get description for current breadth value
  const getBreadthDescription = useCallback((value: number): string => {
    if (value <= 5) return 'Few connections per word';
    if (value <= 15) return 'Moderate connections';
    if (value <= 30) return 'Many connections';
    return 'Maximum connections (may be slow)';
  }, []);

  // Handle depth slider changes with improved performance
  const handleDepthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10);
    setLocalDepth(newValue);
  };

  // Handle breadth slider changes with improved performance
  const handleBreadthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value, 10);
    setLocalBreadth(newValue);
  };

  // Only update parent state when dragging ends
  const handleDepthChangeEnd = () => {
    if (localDepth !== depth) {
      onDepthChange(localDepth);
    }
    setIsDraggingDepth(false);
  };

  const handleBreadthChangeEnd = () => {
    if (localBreadth !== breadth) {
      onBreadthChange(localBreadth);
    }
    setIsDraggingBreadth(false);
  };

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
            onChange={handleDepthChange}
            onMouseDown={() => setIsDraggingDepth(true)}
            onMouseUp={handleDepthChangeEnd}
            onTouchStart={() => setIsDraggingDepth(true)}
            onTouchEnd={handleDepthChangeEnd}
          />
          <div className="slider-description">{getDepthDescription(localDepth)}</div>
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
            onChange={handleBreadthChange}
            onMouseDown={() => setIsDraggingBreadth(true)}
            onMouseUp={handleBreadthChangeEnd}
            onTouchStart={() => setIsDraggingBreadth(true)}
            onTouchEnd={handleBreadthChangeEnd}
          />
          <div className="slider-description">{getBreadthDescription(localBreadth)}</div>
        </div>
      </div>
    </div>
  );
};

export default NetworkControls; 