import React, { useState, useEffect, useCallback } from 'react';
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
  // Local state to handle changes before submitting to parent
  const [localDepth, setLocalDepth] = useState<number>(depth);
  const [localBreadth, setLocalBreadth] = useState<number>(breadth);
  const [isDirty, setIsDirty] = useState<boolean>(false);

  // Update local state when props change
  useEffect(() => {
    setLocalDepth(depth);
  }, [depth]);

  useEffect(() => {
    setLocalBreadth(breadth);
  }, [breadth]);

  // Debounce changes to not overload the API
  const debouncedUpdate = useCallback(() => {
    if (isDirty) {
      console.log(`Updating network with depth=${localDepth}, breadth=${localBreadth}`);
      onDepthChange(localDepth);
      onBreadthChange(localBreadth);
      setIsDirty(false);
    }
  }, [localDepth, localBreadth, isDirty, onDepthChange, onBreadthChange]);

  // Apply changes after a short delay
  useEffect(() => {
    if (isDirty) {
      const timer = setTimeout(() => {
        debouncedUpdate();
      }, 500); // 500ms delay before triggering the update
      
      return () => clearTimeout(timer);
    }
  }, [isDirty, debouncedUpdate]);

  // Safeguard the depth and breadth values
  const safeDepth = Math.max(1, Math.min(5, localDepth));
  const safeBreadth = Math.max(5, Math.min(75, localBreadth));

  // Handle depth change
  const handleDepthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value);
    if (!isNaN(newValue)) {
      setLocalDepth(newValue);
      setIsDirty(true);
    }
  };

  // Handle breadth change
  const handleBreadthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(e.target.value);
    if (!isNaN(newValue)) {
      setLocalBreadth(newValue);
      setIsDirty(true);
    }
  };

  // Handle immediate apply
  const handleApplyChanges = () => {
    if (isDirty) {
      onDepthChange(localDepth);
      onBreadthChange(localBreadth);
      setIsDirty(false);
    }
  };

  return (
    <div className="network-controls">
      <h3>Network Settings</h3>
      
      <div className="sliders-row">
        <div className="slider-container">
          <label htmlFor="depth-slider">Relation Depth: {safeDepth}</label>
          <input
            id="depth-slider"
            type="range"
            min={1}
            max={5}
            step={1}
            value={safeDepth}
            onChange={handleDepthChange}
          />
          <div className="slider-description">
            {safeDepth === 1 && "Only direct relationships"}
            {safeDepth === 2 && "Friends of friends"}
            {safeDepth === 3 && "Extended network"}
            {safeDepth === 4 && "Wide network reach"}
            {safeDepth === 5 && "Maximum depth (slower)"}
          </div>
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
            onChange={handleBreadthChange}
          />
          <div className="slider-description">
            {safeBreadth <= 15 && "Few connections (faster)"}
            {safeBreadth > 15 && safeBreadth <= 30 && "Balanced connections"}
            {safeBreadth > 30 && safeBreadth <= 50 && "Rich network"}
            {safeBreadth > 50 && "Dense network (slower)"}
          </div>
        </div>
      </div>
      
      {isDirty && (
        <div className="controls-actions">
          <button 
            className="apply-button"
            onClick={handleApplyChanges}
            title="Apply changes immediately"
          >
            Apply Changes
          </button>
          <div className="update-message">
            Changes will apply automatically in a moment...
          </div>
        </div>
      )}
    </div>
  );
};

export default NetworkControls; 