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
  onChangeCommitted?: (depth: number, breadth: number) => void;
  className?: string;
}

const ControlContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '4px 6px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.7))',
  borderRadius: '4px',
  backdropFilter: 'blur(4px)',
  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
  width: '100%',
  maxWidth: '100%',
  minHeight: '32px',
  '@media (max-width: 768px)': {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: '2px 4px'
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  minWidth: 0, // Allow container to shrink below minimum content size
  gap: '4px',
  '@media (max-width: 480px)': {
    minWidth: '100%',
    marginBottom: '2px'
  }
}));

const ControlLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  whiteSpace: 'nowrap',
  color: 'var(--text-color)',
  fontSize: '0.7rem',
  fontWeight: 'bold',
  minWidth: '36px',
  padding: '0 2px',
  '& .value': {
    marginLeft: '4px',
    minWidth: '16px'
  },
  '& .MuiSvgIcon-root': {
    fontSize: '0.75rem',
    opacity: 0.7,
    cursor: 'help',
    marginLeft: '2px'
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color)',
  height: 2,
  padding: '6px 0',
  marginRight: '8px',
  '& .MuiSlider-thumb': {
    height: 10,
    width: 10,
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
    fontSize: '0.65rem',
    padding: '1px 4px', 
    backgroundColor: 'var(--primary-color)'
  }
}));

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange,
  onChangeCommitted,
  className
}) => {
  const handleDepthChange = useCallback((_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    onDepthChange(value);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    onBreadthChange(value);
  }, [onBreadthChange]);

  const handleCommitted = useCallback((_event: React.SyntheticEvent | Event, _value: number | number[]) => {
    if (onChangeCommitted) {
      onChangeCommitted(depth, breadth);
    }
  }, [depth, breadth, onChangeCommitted]);

  return (
    <ControlContainer className={className}>
      <SliderContainer>
        <ControlLabel>
          Depth:{depth}
          <Tooltip title="Steps away from main word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
        </ControlLabel>
        <StyledSlider
          value={depth}
          onChange={handleDepthChange}
          onChangeCommitted={handleCommitted}
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
          Breadth:{breadth}
          <Tooltip title="Relations per word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
        </ControlLabel>
        <StyledSlider
          value={breadth}
          onChange={handleBreadthChange}
          onChangeCommitted={handleCommitted}
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