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

const ControlContainer = styled(Box)({
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  padding: '4px 8px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.8))',
  borderRadius: '4px',
  backdropFilter: 'blur(4px)',
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
  maxWidth: 'fit-content',
  '@media (max-width: 600px)': {
    width: '100%',
    maxWidth: '100%'
  }
});

const SliderContainer = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  minWidth: '160px',
  maxWidth: '200px',
  '@media (max-width: 600px)': {
    flex: 1
  }
});

const ControlLabel = styled(Typography)({
  display: 'flex',
  alignItems: 'center',
  gap: '2px',
  color: 'var(--text-color)',
  fontSize: '0.75rem',
  whiteSpace: 'nowrap',
  minWidth: '45px',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '2px',
    minWidth: '16px'
  },
  '& .MuiSvgIcon-root': {
    fontSize: '0.875rem',
    opacity: 0.7,
    cursor: 'help',
    marginLeft: '2px'
  }
});

const StyledSlider = styled(Slider)({
  color: 'var(--primary-color)',
  height: 2,
  padding: '8px 0',
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    backgroundColor: 'var(--primary-color)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 4px var(--primary-color-rgb, rgba(29, 53, 87, 0.16))'
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
    height: 4,
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
    <ControlContainer>
      <SliderContainer>
        <ControlLabel>
          Depth
          <Tooltip title="Steps away from main word" arrow placement="top">
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
          size="small"
          aria-label="Network depth"
        />
      </SliderContainer>

      <SliderContainer>
        <ControlLabel>
          Breadth
          <Tooltip title="Relations per word" arrow placement="top">
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
  );
};

export default NetworkControls; 