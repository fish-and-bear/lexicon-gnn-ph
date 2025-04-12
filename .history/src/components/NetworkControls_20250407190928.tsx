import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';

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
  const handleDepthChange = useCallback((_event: Event, newValue: number | number[]) => {
    onDepthChange(Array.isArray(newValue) ? newValue[0] : newValue);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_event: Event, newValue: number | number[]) => {
    onBreadthChange(Array.isArray(newValue) ? newValue[0] : newValue);
  }, [onBreadthChange]);

  return (
    <div className="controls-container">
      <div className="graph-controls">
        <div className="slider-container">
          <Typography gutterBottom>
            Network Depth: {depth}
          </Typography>
          <Slider
            value={depth}
            onChange={handleDepthChange}
            min={1}
            max={5}
            step={1}
            marks
            valueLabelDisplay="auto"
            aria-label="Network depth"
          />
        </div>
        <div className="slider-container">
          <Typography gutterBottom>
            Relations per Word: {breadth}
          </Typography>
          <Slider
            value={breadth}
            onChange={handleBreadthChange}
            min={5}
            max={50}
            step={5}
            marks
            valueLabelDisplay="auto"
            aria-label="Relations per word"
          />
        </div>
      </div>
    </div>
  );
};

export default NetworkControls; 