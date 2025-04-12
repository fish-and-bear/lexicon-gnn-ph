import React, { useCallback, useState, useEffect, useRef } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import TuneIcon from '@mui/icons-material/Tune';
import { styled, alpha } from '@mui/material/styles';
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
  gap: '12px',
  padding: '8px 12px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.75))',
  borderRadius: '8px',
  backdropFilter: 'blur(8px)',
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.12)',
  width: '100%',
  maxWidth: '100%',
  minHeight: '38px',
  touchAction: 'manipulation',
  userSelect: 'none',
  transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    backgroundColor: 'var(--controls-background-color-hover, rgba(255, 255, 255, 0.9))',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
  },
  '@media (max-width: 768px)': {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: '8px 10px',
    gap: '10px'
  }
}));

const SliderGroupHeader = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  marginBottom: '2px',
  '& svg': {
    fontSize: '16px',
    color: 'var(--primary-color, #1976d2)',
    opacity: 0.7
  }
});

const SliderGroup = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  flex: 1,
  minWidth: 0,
  marginRight: '4px',
  '@media (max-width: 480px)': {
    minWidth: '100%'
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  minWidth: 0,
  gap: '10px',
  position: 'relative',
  padding: '4px 0',
  '@media (max-width: 480px)': {
    marginBottom: '4px'
  }
}));

const ControlLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  whiteSpace: 'nowrap',
  color: 'var(--text-color)',
  fontSize: '0.8rem',
  fontWeight: 'bold',
  minWidth: '44px',
  padding: '0 2px',
  userSelect: 'none',
  '& .MuiSvgIcon-root': {
    fontSize: '0.75rem',
    opacity: 0.7,
    cursor: 'help',
    marginLeft: '4px'
  }
}));

const ValueDisplay = styled(Box)(({ theme }) => ({
  minWidth: '28px',
  height: '20px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '0.75rem',
  fontWeight: 'bold',
  color: 'var(--primary-color, #1976d2)',
  borderRadius: '4px',
  padding: '0 4px',
  backgroundColor: alpha('#1976d2', 0.1),
  border: '1px solid',
  borderColor: alpha('#1976d2', 0.2),
  transition: 'all 0.2s ease'
}));

const ValueBubble = styled('div')<{ active?: boolean }>(({ active }) => ({
  position: 'absolute',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  top: '-28px',
  transform: 'translateX(-50%) scale(' + (active ? '1' : '0.8') + ')',
  minWidth: '32px',
  height: '24px',
  padding: '0 6px',
  backgroundColor: 'var(--primary-color, #1976d2)',
  color: '#fff',
  fontWeight: 'bold',
  fontSize: '0.8rem',
  borderRadius: '4px',
  opacity: active ? 1 : 0,
  pointerEvents: 'none',
  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
  transition: 'opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1), transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
  zIndex: 5,
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-6px',
    left: '50%',
    transform: 'translateX(-50%)',
    borderLeft: '6px solid transparent',
    borderRight: '6px solid transparent',
    borderTop: '6px solid var(--primary-color, #1976d2)'
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: 'var(--primary-color, #1976d2)',
  height: 4,
  padding: '13px 0',
  marginRight: '8px',
  cursor: 'pointer',
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
    marginTop: -6,
    marginLeft: -8,
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0 0 0 8px ${alpha('#1976d2', 0.16)}`,
      transform: 'scale(1.15)'
    },
    '&.Mui-active': {
      boxShadow: `0 0 0 12px ${alpha('#1976d2', 0.24)}`,
      transform: 'scale(1.25)'
    },
    '&::before': {
      display: 'none'
    },
    '@media (pointer: coarse)': {
      height: 20,
      width: 20,
      marginTop: -8,
      marginLeft: -10
    }
  },
  '& .MuiSlider-track': {
    height: 4,
    borderRadius: 2,
    background: 'linear-gradient(90deg, rgba(29, 53, 87, 0.6) 0%, #1976d2 100%)'
  },
  '& .MuiSlider-rail': {
    height: 4,
    borderRadius: 2,
    opacity: 0.3,
    backgroundColor: 'var(--slider-rail-color, #bdbdbd)'
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--text-color)',
    height: 8,
    width: 2,
    marginTop: -2,
    opacity: 0.4
  },
  '& .MuiSlider-markLabel': {
    fontSize: '0.65rem',
    fontWeight: 'bold',
    top: '20px',
    color: 'var(--text-light-color, #757575)',
    '&.MuiSlider-markLabelActive': {
      color: 'var(--text-color)'
    }
  },
  '& .MuiSlider-markActive': {
    backgroundColor: 'currentColor',
    opacity: 0.8
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
  const [animateIn, setAnimateIn] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set initial bubble positions
    const depthPercent = ((depth - 1) / 3) * 100;
    const breadthPercent = ((breadth - 5) / 45) * 100;
    setDepthBubbleLeft(depthPercent);
    setBreadthBubbleLeft(breadthPercent);
    
    // Trigger entrance animation
    setTimeout(() => setAnimateIn(true), 50);
  }, [depth, breadth]);

  const depthMarks = [
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4' }
  ];

  const breadthMarks = [
    { value: 5, label: '5' },
    { value: 20, label: '20' },
    { value: 35, label: '35' },
    { value: 50, label: '50' }
  ];

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

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    // Add keyboard shortcuts for quick adjustments
    if (document.activeElement === containerRef.current || 
        containerRef.current?.contains(document.activeElement)) {
      
      if (e.key === 'ArrowUp' && depth < 4) {
        onDepthChange(depth + 1);
        e.preventDefault();
      } else if (e.key === 'ArrowDown' && depth > 1) {
        onDepthChange(depth - 1);
        e.preventDefault();
      } else if (e.key === 'ArrowRight' && breadth < 50) {
        onBreadthChange(Math.min(breadth + 5, 50));
        e.preventDefault();
      } else if (e.key === 'ArrowLeft' && breadth > 5) {
        onBreadthChange(Math.max(breadth - 5, 5));
        e.preventDefault();
      }
    }
  }, [depth, breadth, onDepthChange, onBreadthChange]);

  return (
    <ControlContainer 
      ref={containerRef}
      className={className} 
      role="group" 
      aria-label="Network visualization controls"
      onKeyDown={handleKeyDown}
      tabIndex={0}
      sx={{
        opacity: animateIn ? 1 : 0,
        transform: animateIn ? 'translateY(0)' : 'translateY(10px)'
      }}
    >
      <SliderGroup>
        <SliderGroupHeader>
          <TuneIcon />
          <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
            Network Controls
          </Typography>
        </SliderGroupHeader>
        
        <SliderContainer>
          <ControlLabel id="depth-slider-label">
            Depth
            <Tooltip title="Number of steps away from the main word" arrow placement="top">
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
              onMouseLeave={() => setIsDepthActive(false)}
              onTouchStart={() => setIsDepthActive(true)}
              onTouchEnd={() => setIsDepthActive(false)}
              min={1}
              max={4}
              step={1}
              marks={depthMarks}
              size="medium"
              aria-labelledby="depth-slider-label"
              aria-valuetext={`Depth: ${depth}`}
            />
          </Box>
          
          <ValueDisplay>{depth}</ValueDisplay>
        </SliderContainer>

        <SliderContainer>
          <ControlLabel id="breadth-slider-label">
            Breadth
            <Tooltip title="Maximum number of relations per word" arrow placement="top">
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
              onMouseLeave={() => setIsBreadthActive(false)}
              onTouchStart={() => setIsBreadthActive(true)}
              onTouchEnd={() => setIsBreadthActive(false)}
              min={5}
              max={50}
              step={5}
              marks={breadthMarks}
              size="medium"
              aria-labelledby="breadth-slider-label"
              aria-valuetext={`Breadth: ${breadth}`}
            />
          </Box>
          
          <ValueDisplay>{breadth}</ValueDisplay>
        </SliderContainer>
      </SliderGroup>
    </ControlContainer>
  );
};

export default React.memo(NetworkControls); 