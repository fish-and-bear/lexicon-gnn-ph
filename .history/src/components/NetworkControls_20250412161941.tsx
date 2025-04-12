import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import RestartAltIcon from '@mui/icons-material/RestartAlt';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
}

const ControlsContainer = styled(Box)({
  position: 'absolute',
  bottom: '15px',
  left: '15px',
  right: '15px',
  display: 'flex',
  justifyContent: 'center',
  zIndex: 100,
  width: 'auto',
});

const ControlsCard = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '20px',
  padding: '12px 16px',
  backgroundColor: 'var(--card-bg-color, rgba(255, 255, 255, 0.9))',
  backdropFilter: 'blur(10px)',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
  borderRadius: '12px',
  maxWidth: '800px',
  width: 'auto',
  '@media (max-width: 600px)': {
    flexDirection: 'column',
    gap: '12px',
    padding: '12px',
    width: '100%',
  }
});

const ZoomControls = styled(Box)({
  display: 'flex',
  gap: '8px',
  borderRight: '1px solid var(--border-color, rgba(0, 0, 0, 0.1))',
  paddingRight: '20px',
  '@media (max-width: 600px)': {
    borderRight: 'none',
    borderBottom: '1px solid var(--border-color, rgba(0, 0, 0, 0.1))',
    paddingRight: 0,
    paddingBottom: '12px',
    width: '100%',
    justifyContent: 'center',
  }
});

const SliderGroup = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
  flex: 1,
  minWidth: '120px',
});

const SliderHeader = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
});

const SliderLabel = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  color: 'var(--text-color, #333)',
  fontSize: '0.85rem',
  fontWeight: 500,
});

const SliderValue = styled(Box)({
  fontSize: '0.85rem',
  fontWeight: 600,
  color: 'var(--accent-color, #3a86ff)',
  background: 'var(--slider-value-bg, rgba(58, 134, 255, 0.1))',
  padding: '2px 8px',
  borderRadius: '10px',
  minWidth: '24px',
  textAlign: 'center',
});

const StyledSlider = styled(Slider)({
  color: 'var(--accent-color, #3a86ff)',
  height: 4,
  padding: '13px 0',
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: '0 0 0 8px var(--accent-color-rgb, rgba(58, 134, 255, 0.16))'
    },
    '&.Mui-active': {
      boxShadow: '0 0 0 12px var(--accent-color-rgb, rgba(58, 134, 255, 0.16))'
    }
  },
  '& .MuiSlider-track': {
    height: 4,
    borderRadius: 2,
  },
  '& .MuiSlider-rail': {
    height: 4,
    borderRadius: 2,
    opacity: 0.2,
    backgroundColor: 'var(--text-color, #333)',
  },
  '& .MuiSlider-mark': {
    backgroundColor: 'var(--accent-color, #3a86ff)',
    height: 8,
    width: 2,
    marginTop: -2,
    opacity: 0.4,
  },
  '& .MuiSlider-markActive': {
    opacity: 0.7,
    backgroundColor: 'currentColor',
  },
  '& .MuiSlider-markLabel': {
    fontSize: '0.7rem',
    color: 'var(--text-color, #333)',
    opacity: 0.7,
  }
});

const IconButton = styled('button')({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: '34px',
  height: '34px',
  backgroundColor: 'var(--button-bg-color, rgba(58, 134, 255, 0.1))',
  color: 'var(--accent-color, #3a86ff)',
  border: 'none',
  borderRadius: '10px',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: 'var(--button-hover-bg-color, rgba(58, 134, 255, 0.2))',
    transform: 'translateY(-1px)',
  },
  '&:active': {
    transform: 'translateY(1px)',
  },
  '& svg': {
    fontSize: '18px'
  }
});

const StyledInfoIcon = styled(InfoIcon)({
  fontSize: '16px',
  opacity: 0.7,
  cursor: 'help',
  color: 'var(--text-color, rgba(0, 0, 0, 0.6))',
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
    <ControlsContainer>
      <ControlsCard>
        <ZoomControls>
          <IconButton onClick={onZoomIn} title="Zoom In">
            <ZoomInIcon />
          </IconButton>
          <IconButton onClick={onZoomOut} title="Zoom Out">
            <ZoomOutIcon />
          </IconButton>
          <IconButton onClick={onResetZoom} title="Reset View">
            <RestartAltIcon />
          </IconButton>
        </ZoomControls>
        
        <SliderGroup>
          <SliderHeader>
            <SliderLabel>
              Depth
              <Tooltip title="Controls how many steps away from the main word to explore. Higher values show more distant connections." arrow placement="top">
                <StyledInfoIcon />
              </Tooltip>
            </SliderLabel>
            <SliderValue>{depth}</SliderValue>
          </SliderHeader>
          <StyledSlider
            value={depth}
            onChange={handleDepthChange}
            min={1}
            max={4}
            step={1}
            marks={[
              { value: 1, label: '1' },
              { value: 2 },
              { value: 3 },
              { value: 4, label: '4' }
            ]}
            aria-label="Network depth - steps away from main word"
          />
        </SliderGroup>

        <SliderGroup>
          <SliderHeader>
            <SliderLabel>
              Breadth
              <Tooltip title="Controls how many related words to show for each node. Higher values show more connections per word." arrow placement="top">
                <StyledInfoIcon />
              </Tooltip>
            </SliderLabel>
            <SliderValue>{breadth}</SliderValue>
          </SliderHeader>
          <StyledSlider
            value={breadth}
            onChange={handleBreadthChange}
            min={5}
            max={50}
            step={5}
            marks={[
              { value: 5, label: '5' },
              { value: 20 },
              { value: 35 },
              { value: 50, label: '50' }
            ]}
            aria-label="Network breadth - connections per word"
          />
        </SliderGroup>
      </ControlsCard>
    </ControlsContainer>
  );
};

export default NetworkControls; 