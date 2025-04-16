import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import { alpha } from '@mui/material/styles';

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
  flexDirection: 'column',
  gap: '8px',
  padding: '16px',
  margin: '0 0 16px 0', 
  backgroundColor: alpha(theme.palette.background.paper, 0.7),
  borderRadius: theme.shape.borderRadius,
  boxShadow: `0 1px 3px ${alpha(theme.palette.text.primary, 0.1)}`,
  border: `1px solid ${alpha(theme.palette.divider, 0.3)}`,
  color: theme.palette.text.primary,
  transition: 'all 0.2s ease',
  
  '&:hover': {
    backgroundColor: theme.palette.background.paper,
    boxShadow: `0 2px 5px ${alpha(theme.palette.text.primary, 0.15)}`,
  },

  '.dark &': {
    backgroundColor: 'var(--graph-bg-color)',
    color: 'var(--text-color)',
    border: 'none !important',
    boxShadow: 'none !important',
    
    '&:hover': {
      backgroundColor: 'var(--card-bg-color-elevated)',
      boxShadow: 'none !important',
      border: 'none !important',
    },
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  minWidth: 0,
  gap: '8px',
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
  minWidth: '55px',
}));

const ValueDisplay = styled('span')(({ theme }) => ({
  display: 'inline-block',
  minWidth: '16px',
  marginLeft: '4px',
  textAlign: 'center',
  fontWeight: 'bold',
  fontSize: '0.8rem',
  color: 'var(--primary-color, #3d5a80)',
  transition: 'all 0.15s ease',
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.primary.main,
  height: 6,
  padding: '16px 0',
  '& .MuiSlider-rail': {
    height: 6,
    backgroundColor: alpha(theme.palette.text.primary, 0.1),
    opacity: 1,
  },
  '& .MuiSlider-track': {
    height: 6,
  },
  '& .MuiSlider-thumb': {
    width: 14,
    height: 14,
    boxShadow: `0 2px 5px ${alpha(theme.palette.text.primary, 0.2)}`,
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0 3px 7px ${alpha(theme.palette.text.primary, 0.3)}`,
    },
  },

  '.dark &': {
    color: 'var(--primary-color)',
    boxShadow: 'none !important',
    
    '& .MuiSlider-rail': {
      backgroundColor: 'rgba(255, 255, 255, 0.2)',
      boxShadow: 'none !important',
      border: 'none !important',
    },
    '& .MuiSlider-track': {
      backgroundColor: 'var(--primary-color)',
      boxShadow: 'none !important',
      border: 'none !important',
    },
    '& .MuiSlider-thumb': {
      backgroundColor: 'var(--primary-color)',
      boxShadow: 'none !important',
      border: 'none !important',
      '&:hover, &.Mui-focusVisible, &.Mui-active, &:focus': {
        boxShadow: 'none !important',
        backgroundColor: 'var(--gold)',
        border: 'none !important',
        outline: 'none !important',
      },
    },
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
          Depth: {depth}
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
          Breadth: {breadth}
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