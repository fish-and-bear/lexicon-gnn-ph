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
  padding: '6px 8px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.75))',
  borderRadius: '4px',
  backdropFilter: 'blur(6px)',
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
  width: '100%',
  maxWidth: '100%',
  minHeight: '36px',
  '@media (max-width: 768px)': {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: '6px',
    padding: '4px 6px'
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  minWidth: 0,
  gap: '6px',
  height: '24px',
  '@media (max-width: 480px)': {
    minWidth: '100%',
    marginBottom: '4px'
  }
}));

const ControlLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  whiteSpace: 'nowrap',
  color: 'var(--text-color, #333)',
  fontSize: '0.75rem',
  fontWeight: '600',
  minWidth: '60px',
  padding: '0 2px',
  height: '24px',
  '& .MuiSvgIcon-root': {
    fontSize: '0.85rem',
    opacity: 0.7,
    cursor: 'help',
    marginLeft: '4px',
    verticalAlign: 'middle'
  }
}));

const ValueDisplay = styled('span')(({ theme }) => ({
  display: 'inline-block',
  minWidth: '16px',
  marginLeft: '4px',
  textAlign: 'center',
  fontWeight: 'bold',
  fontSize: '0.8rem',
  color: 'var(--primary-color, #3d5a80)',
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color, #3d5a80)',
  height: 3,
  padding: '8px 0',
  marginRight: '4px',
  flex: 1,
  minWidth: 0,
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    backgroundColor: 'var(--primary-color, #3d5a80)',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 5px var(--primary-color-rgb, rgba(61, 90, 128, 0.16))'
    },
    '&:active': {
      boxShadow: '0 0 0 8px var(--primary-color-rgb, rgba(61, 90, 128, 0.24))'
    }
  },
  '& .MuiSlider-track': {
    height: 3,
    border: 'none',
  },
  '& .MuiSlider-rail': {
    height: 3,
    opacity: 0.3,
    backgroundColor: 'var(--text-color-light, #adb5bd)',
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--text-color, #333)',
    height: 4,
    width: 1,
    marginTop: -1,
    opacity: 0.4
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor',
    opacity: 0.7
  },
  '& .MuiSlider-valueLabel': {
    fontSize: '0.7rem',
    padding: '2px 6px',
    borderRadius: '4px',
    backgroundColor: 'var(--primary-color, #3d5a80)',
    fontWeight: 'bold',
    top: -6,
    '&:before': {
      display: 'none'
    }
  },
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
          Depth
          <Tooltip title="Steps away from main word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
          <ValueDisplay>{depth}</ValueDisplay>
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
          Breadth
          <Tooltip title="Relations per word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
          <ValueDisplay>{breadth}</ValueDisplay>
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