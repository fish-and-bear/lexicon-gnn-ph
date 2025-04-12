import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

// Custom styled components
const ControlContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: '16px',
  alignItems: 'center',
  padding: '8px 12px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.9))',
  borderRadius: '8px',
  backdropFilter: 'blur(8px)',
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
  '@media (max-width: 600px)': {
    flexDirection: 'column',
    gap: '8px',
    width: '100%'
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  minWidth: '200px',
  flex: 1,
  '@media (max-width: 600px)': {
    width: '100%'
  }
}));

const ControlLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  color: 'var(--text-color)',
  minWidth: '60px',
  fontSize: '0.875rem',
  whiteSpace: 'nowrap',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '4px'
  },
  '& .MuiSvgIcon-root': {
    fontSize: '1rem',
    opacity: 0.7,
    cursor: 'help'
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color)',
  height: 4,
  width: '100%',
  padding: '13px 0',
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: 'var(--primary-color)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 6px var(--primary-color-rgb, rgba(29, 53, 87, 0.16))'
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
    backgroundColor: 'var(--text-color)',
    height: 8,
    width: 1,
    opacity: 0.3
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor'
  },
  '& .MuiSlider-valueLabel': {
    fontSize: '0.75rem',
    padding: '2px 4px',
    backgroundColor: 'var(--primary-color)'
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
    <ControlContainer>
      <SliderContainer>
        <ControlLabel>
          Depth
          <Tooltip title="Controls how many steps away from the main word to explore" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
          <span className="value">{depth}</span>
        </ControlLabel>
        <StyledSlider
          value={depth}
          onChange={handleDepthChange}
          min={1}
          max={4}
          step={1}
          marks
          valueLabelDisplay="auto"
          aria-label="Network depth"
        />
      </SliderContainer>

      <SliderContainer>
        <ControlLabel>
          Breadth
          <Tooltip title="Controls how many relationships to show per word" arrow placement="top">
            <InfoIcon fontSize="small" />
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
      </SliderContainer>
    </ControlContainer>
  );
};

export default NetworkControls; 