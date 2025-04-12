import React, { useCallback, useState } from 'react';
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
  backdropFilter: 'blur(4px)',
  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
  width: '100%',
  maxWidth: '100%',
  minHeight: '32px',
  touchAction: 'manipulation',
  userSelect: 'none',
  transition: 'background-color 0.2s ease',
  '&:hover': {
    backgroundColor: 'var(--controls-background-color-hover, rgba(255, 255, 255, 0.85))'
  },
  '@media (max-width: 768px)': {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: '4px 6px'
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  minWidth: 0,
  gap: '4px',
  position: 'relative',
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
  fontSize: '0.75rem',
  fontWeight: 'bold',
  minWidth: '38px',
  padding: '0 2px',
  userSelect: 'none',
  '& .MuiSvgIcon-root': {
    fontSize: '0.75rem',
    opacity: 0.7,
    cursor: 'help',
    marginLeft: '2px'
  }
}));

const ValueBubble = styled('div')<{ active?: boolean }>(({ active }) => ({
  position: 'absolute',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  top: '-22px',
  transform: 'translateX(-50%)',
  minWidth: '24px',
  height: '20px',
  padding: '0 4px',
  backgroundColor: 'var(--primary-color, #1976d2)',
  color: '#fff',
  fontWeight: 'bold',
  fontSize: '0.7rem',
  borderRadius: '3px',
  opacity: active ? 1 : 0,
  transition: 'opacity 0.2s ease',
  pointerEvents: 'none',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-4px',
    left: '50%',
    transform: 'translateX(-50%)',
    borderLeft: '4px solid transparent',
    borderRight: '4px solid transparent',
    borderTop: '4px solid var(--primary-color, #1976d2)'
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color, #1976d2)',
  height: 3,
  padding: '8px 0',
  marginRight: '8px',
  cursor: 'pointer',
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    backgroundColor: 'var(--primary-color, #1976d2)',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
    marginTop: -4.5,
    marginLeft: -6,
    transition: 'box-shadow 0.2s ease-in-out, transform 0.15s ease',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 6px var(--primary-color-rgb, rgba(29, 53, 87, 0.2))',
      transform: 'scale(1.15)'
    },
    '&.Mui-active': {
      boxShadow: '0 0 0 8px var(--primary-color-rgb, rgba(29, 53, 87, 0.3))',
      transform: 'scale(1.2)'
    }
  },
  '& .MuiSlider-track': {
    height: 3,
    borderRadius: 2
  },
  '& .MuiSlider-rail': {
    height: 3,
    borderRadius: 2,
    opacity: 0.25
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--text-color)',
    height: 6,
    width: 1,
    marginTop: -1.5,
    opacity: 0.4
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor',
    opacity: 0.7
  },
  '& .MuiSlider-valueLabel': {
    display: 'none'
  },
  '&.Mui-disabled': {
    opacity: 0.5,
    cursor: 'not-allowed'
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
  const [isDepthActive, setIsDepthActive] = useState(false);
  const [isBreadthActive, setIsBreadthActive] = useState(false);
  const [depthBubbleLeft, setDepthBubbleLeft] = useState(0);
  const [breadthBubbleLeft, setBreadthBubbleLeft] = useState(0);

  const handleDepthChange = useCallback((_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    onDepthChange(value);
    
    // Update bubble position
    const percent = ((value - 1) / 3) * 100;
    setDepthBubbleLeft(percent);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    onBreadthChange(value);
    
    // Update bubble position
    const percent = ((value - 5) / 45) * 100;
    setBreadthBubbleLeft(percent);
  }, [onBreadthChange]);

  const handleCommitted = useCallback((_event: React.SyntheticEvent | Event, _value: number | number[]) => {
    if (onChangeCommitted) {
      onChangeCommitted(depth, breadth);
    }
  }, [depth, breadth, onChangeCommitted]);

  return (
    <ControlContainer className={className} role="group" aria-label="Network visualization controls">
      <SliderContainer>
        <ControlLabel id="depth-slider-label">
          Depth:{depth}
          <Tooltip title="Steps away from main word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
        </ControlLabel>
        <Box sx={{ position: 'relative', flex: 1 }}>
          <ValueBubble 
            active={isDepthActive} 
            style={{ left: `${depthBubbleLeft}%` }}
          >
            {depth}
          </ValueBubble>
          <StyledSlider
            value={depth}
            onChange={handleDepthChange}
            onChangeCommitted={handleCommitted}
            onMouseDown={() => setIsDepthActive(true)}
            onMouseUp={() => setIsDepthActive(false)}
            onTouchStart={() => setIsDepthActive(true)}
            onTouchEnd={() => setIsDepthActive(false)}
            min={1}
            max={4}
            step={1}
            marks
            size="small"
            aria-labelledby="depth-slider-label"
            aria-valuetext={`Depth: ${depth}`}
          />
        </Box>
      </SliderContainer>

      <SliderContainer>
        <ControlLabel id="breadth-slider-label">
          Breadth:{breadth}
          <Tooltip title="Relations per word" arrow placement="top">
            <InfoIcon fontSize="small" />
          </Tooltip>
        </ControlLabel>
        <Box sx={{ position: 'relative', flex: 1 }}>
          <ValueBubble 
            active={isBreadthActive} 
            style={{ left: `${breadthBubbleLeft}%` }}
          >
            {breadth}
          </ValueBubble>
          <StyledSlider
            value={breadth}
            onChange={handleBreadthChange}
            onChangeCommitted={handleCommitted}
            onMouseDown={() => setIsBreadthActive(true)}
            onMouseUp={() => setIsBreadthActive(false)}
            onTouchStart={() => setIsBreadthActive(true)}
            onTouchEnd={() => setIsBreadthActive(false)}
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
            aria-labelledby="breadth-slider-label"
            aria-valuetext={`Breadth: ${breadth}`}
          />
        </Box>
      </SliderContainer>
    </ControlContainer>
  );
};

export default React.memo(NetworkControls); 