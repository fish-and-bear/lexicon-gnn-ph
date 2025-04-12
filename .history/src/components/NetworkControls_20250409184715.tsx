import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import { styled } from '@mui/material/styles';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

// Custom styled components
const ControlLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '4px',
  color: 'var(--text-color)',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: 'auto',
  },
  '& .MuiSvgIcon-root': {
    fontSize: '1rem',
    opacity: 0.7,
    cursor: 'help',
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color)',
  '& .MuiSlider-thumb': {
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 8px var(--primary-color-rgb, rgba(29, 53, 87, 0.16))'
    }
  },
  '& .MuiSlider-track': {
    height: 4
  },
  '& .MuiSlider-rail': {
    opacity: 0.3
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--text-color)',
    opacity: 0.3
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor'
  }
}));

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
          <ControlLabel>
            Network Depth
            <Tooltip title="Controls how many steps away from the main word to explore. Higher values show more distant connections but may increase complexity." arrow>
              <InfoIcon />
            </Tooltip>
            <span className="value">{depth}</span>
          </ControlLabel>
          <StyledSlider
            value={depth}
            onChange={handleDepthChange}
            min={1}
            max={5}
            step={1}
            marks={[
              { value: 1, label: '1' },
              { value: 2, label: '2' },
              { value: 3, label: '3' },
              { value: 4, label: '4' },
              { value: 5, label: '5' }
            ]}
            valueLabelDisplay="auto"
            aria-label="Network depth"
          />
        </div>
        <div className="slider-container">
          <ControlLabel>
            Relations per Word
            <Tooltip title="Controls how many relationships to show for each word. Higher values show more connections but may increase visual complexity." arrow>
              <InfoIcon />
            </Tooltip>
            <span className="value">{breadth}</span>
          </ControlLabel>
          <StyledSlider
            value={breadth}
            onChange={handleBreadthChange}
            min={5}
            max={50}
            step={5}
            marks={[
              { value: 5, label: '5' },
              { value: 20, label: '20' },
              { value: 35, label: '35' },
              { value: 50, label: '50' }
            ]}
            valueLabelDisplay="auto"
            aria-label="Relations per word"
          />
        </div>
      </div>
    </div>
  );
};

export default NetworkControls; 