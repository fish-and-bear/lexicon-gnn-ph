import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import TuneIcon from '@mui/icons-material/Tune';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

const ControlContainer = styled(Box)({
  display: 'inline-flex',
  alignItems: 'center',
  gap: '4px',
  padding: '2px 6px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.8))',
  borderRadius: '4px',
  backdropFilter: 'blur(4px)',
  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
  maxWidth: 'fit-content',
  height: '28px',
  '@media (max-width: 600px)': {
    width: 'auto',
    maxWidth: '100%'
  }
});

const SliderContainer = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '2px',
  minWidth: '120px',
  maxWidth: '140px',
  position: 'relative',
  '@media (max-width: 600px)': {
    minWidth: '100px'
  }
});

const ControlLabel = styled(Typography)({
  display: 'flex',
  alignItems: 'center',
  color: 'var(--text-color)',
  fontSize: '0.7rem',
  whiteSpace: 'nowrap',
  minWidth: '32px',
  opacity: 0.8,
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '2px',
    minWidth: '14px'
  }
});

const StyledSlider = styled(Slider)({
  color: 'var(--primary-color)',
  height: 2,
  padding: '6px 0',
  '& .MuiSlider-thumb': {
    height: 10,
    width: 10,
    backgroundColor: 'var(--primary-color)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 4px var(--primary-color-rgb, rgba(29, 53, 87, 0.12))'
    }
  },
  '& .MuiSlider-track': {
    height: 2
  },
  '& .MuiSlider-rail': {
    height: 2,
    opacity: 0.2
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--text-color)',
    height: 3,
    width: 1,
    opacity: 0.2
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor'
  },
  '& .MuiSlider-valueLabel': {
    display: 'none'
  }
});

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
    <Tooltip title="Network depth and breadth controls" placement="top">
      <ControlContainer>
        <TuneIcon sx={{ fontSize: '14px', opacity: 0.6, mr: '2px' }} />
        <SliderContainer>
          <ControlLabel>
            D<span className="value">{depth}</span>
          </ControlLabel>
          <StyledSlider
            value={depth}
            onChange={handleDepthChange}
            min={1}
            max={4}
            step={1}
            marks
            size="small"
            aria-label="Network depth"
          />
        </SliderContainer>

        <SliderContainer>
          <ControlLabel>
            B<span className="value">{breadth}</span>
          </ControlLabel>
          <StyledSlider
            value={breadth}
            onChange={handleBreadthChange}
            min={5}
            max={50}
            step={5}
            marks={[
              { value: 5 },
              { value: 20 },
              { value: 35 },
              { value: 50 }
            ]}
            size="small"
            aria-label="Relations per word"
          />
        </SliderContainer>
      </ControlContainer>
    </Tooltip>
  );
};

export default NetworkControls; 