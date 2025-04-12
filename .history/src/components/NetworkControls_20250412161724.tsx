import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
}

const ControlsBar = styled(Box)({
  display: 'flex',
  width: 'calc(100% - 140px)',
  gap: '16px',
  padding: '6px 8px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.8))',
  borderRadius: '4px',
  alignItems: 'center',
  height: '28px',
  '@media (max-width: 600px)': {
    width: '100%',
    gap: '8px'
  }
});

const SliderGroup = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  flex: 1,
  height: '100%'
});

const Label = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '3px',
  color: 'var(--text-color)',
  fontSize: '0.8rem',
  whiteSpace: 'nowrap',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '3px',
    minWidth: '14px'
  }
});

const NetworkSlider = styled(Slider)({
  color: 'var(--accent-color, #3a86ff)',
  padding: '6px 0',
  height: 4,
  '& .MuiSlider-thumb': {
    height: 14,
    width: 14,
    marginTop: -5,
    '&:hover': {
      boxShadow: '0 0 0 6px var(--accent-color-rgb, rgba(58, 134, 255, 0.16))'
    }
  },
  '& .MuiSlider-track': {
    height: 4
  },
  '& .MuiSlider-rail': {
    height: 4,
    opacity: 0.3
  },
  '& .MuiSlider-mark': {
    height: 6,
    width: 1,
    opacity: 0.4
  },
  '& .MuiSlider-markLabel': {
    fontSize: '0.7rem'
  }
});

const StyledInfoIcon = styled(InfoIcon)({
  fontSize: '0.8rem',
  opacity: 0.7,
  cursor: 'help',
  marginTop: '-1px',
  width: '14px',
  height: '14px'
});

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange,
  onZoomIn,
  onZoomOut,
  onResetZoom
}) => {
  const handleDepthChange = useCallback((_: Event, value: number | number[]) => {
    onDepthChange(Array.isArray(value) ? value[0] : value);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_: Event, value: number | number[]) => {
    onBreadthChange(Array.isArray(value) ? value[0] : value);
  }, [onBreadthChange]);

  return (
    <ControlsBar>
      <div className="zoom-controls">
        <button onClick={onZoomIn} className="zoom-button" title="Zoom In">+</button>
        <button onClick={onZoomOut} className="zoom-button" title="Zoom Out">-</button>
        <button onClick={onResetZoom} className="zoom-button" title="Reset View">Reset</button>
      </div>
      
      <SliderGroup>
        <Label>
          Depth
          <Tooltip title="Controls how many steps away from the main word to explore. Higher values show more distant connections." arrow placement="top">
            <StyledInfoIcon />
          </Tooltip>
          <span className="value">{depth}</span>
        </Label>
        <NetworkSlider
          value={depth}
          onChange={handleDepthChange}
          min={1}
          max={4}
          step={1}
          marks
          size="small"
          aria-label="Network depth - steps away from main word"
        />
      </SliderGroup>

      <SliderGroup>
        <Label>
          Breadth
          <Tooltip title="Controls how many related words to show for each node. Higher values show more connections per word." arrow placement="top">
            <StyledInfoIcon />
          </Tooltip>
          <span className="value">{breadth}</span>
        </Label>
        <NetworkSlider
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
          size="small"
          aria-label="Network breadth - connections per word"
        />
      </SliderGroup>
    </ControlsBar>
  );
};

export default NetworkControls; 