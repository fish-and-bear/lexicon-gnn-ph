import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
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

// Styled components copied exactly from old_src_2
const ControlContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
  padding: '6px 8px',
  backgroundColor: 'transparent',
  borderRadius: '4px',
  backdropFilter: 'none',
  boxShadow: 'none',
  width: '100%',
  maxWidth: '100%',
  minHeight: '36px',
  transition: 'all 0.2s ease',
  '.dark &': {
    backgroundColor: 'transparent',
    backdropFilter: 'none',
    border: 'none',
  },
  '&:hover': {
    boxShadow: 'none',
    '.dark &': {
      backgroundColor: 'transparent',
      border: 'none',
    },
    ':not(.dark) &': {
      backgroundColor: 'transparent',
    }
  },
  '@media (max-width: 768px)': {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: '6px',
    padding: '4px 6px'
  }
}));

const SliderContainer = styled(Box)(({/* theme */}) => ({
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

const ControlLabel = styled(Typography)(({/* theme */}) => ({
  display: 'flex',
  alignItems: 'center',
  whiteSpace: 'nowrap',
  color: 'var(--text-color)',
  fontSize: '0.75rem',
  fontWeight: '600',
  minWidth: '55px',
  fontFamily: 'system-ui, -apple-system, sans-serif',
}));

const StyledSlider = styled(Slider)(({/* theme */}) => ({
  color: 'var(--primary-color)',
  height: 3,
  padding: '8px 0',
  marginRight: '4px',
  flex: 1,
  minWidth: 0,
  transition: 'opacity 0.2s ease',
  '.dark &': {
    // No need to override main color here, it's inherited via CSS variable
  },
  '&:hover': {
    opacity: 0.95,
  },
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    backgroundColor: 'currentColor',
    boxShadow: 'none',
    transition: 'all 0.2s ease-out',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: 'none',
    },
    '&:active': {
      boxShadow: 'none'
    }
  },
  '& .MuiSlider-track': {
    height: 3,
    border: 'none',
    backgroundColor: 'var(--accent-color)',
    '.dark &': {
      backgroundColor: 'currentColor',
    }
  },
  '& .MuiSlider-rail': {
    height: 3,
    opacity: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
    '.dark &': {
      backgroundColor: 'rgba(255, 255, 255, 0.15)',
      border: 'none',
    }
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    height: 4,
    width: 1,
    marginTop: -1,
    '.dark &': {
      backgroundColor: 'rgba(255, 255, 255, 0.3)',
      boxShadow: 'none',
    }
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'var(--accent-color)',
    opacity: 1,
    '.dark &': {
      backgroundColor: 'currentColor',
      opacity: 0.8,
      boxShadow: 'none',
    }
  },
  '& .MuiSlider-valueLabel': {
    fontSize: '0.7rem',
    padding: '2px 6px',
    borderRadius: '4px',
    backgroundColor: 'var(--primary-color)',
    color: 'var(--button-text-color)',
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