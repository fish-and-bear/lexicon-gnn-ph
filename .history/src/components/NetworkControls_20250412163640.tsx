import React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

// Props definition for the component
interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (value: number) => void;
  onBreadthChange: (value: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  onCenterMainNode: () => void;
}

// Styled component for the main container
const ControlsContainer = styled(Box)(({ theme }) => ({
  background: theme.palette.mode === 'dark' 
    ? 'rgba(30, 30, 30, 0.9)' 
    : 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(10px)',
  borderRadius: '10px',
  padding: '14px 20px',
  boxShadow: theme.palette.mode === 'dark' 
    ? '0 4px 20px rgba(0, 0, 0, 0.6)' 
    : '0 4px 20px rgba(0, 0, 0, 0.15)',
  border: `1px solid ${theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.12)' 
    : 'rgba(0, 0, 0, 0.07)'}`,
  width: '100%',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: theme.palette.mode === 'dark' 
      ? '0 6px 25px rgba(0, 0, 0, 0.7)' 
      : '0 6px 25px rgba(0, 0, 0, 0.2)',
  },
  display: 'flex',
  flexDirection: 'column',
  gap: '16px',
  '@media (max-width: 600px)': {
    padding: '10px 14px',
    gap: '10px',
  }
}));

// Styled component for the primary controls row
const ControlsRow = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '24px',
  '@media (max-width: 600px)': {
    flexDirection: 'column',
    alignItems: 'stretch',
    gap: '14px'
  }
}));

// Styled component for a single slider
const SliderContainer = styled(Box)(({ theme }) => ({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
}));

// Styled component for button toolbar
const ButtonsToolbar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-end',
  gap: '8px',
  marginLeft: 'auto',
  '@media (max-width: 600px)': {
    marginTop: '8px',
    justifyContent: 'center',
    marginLeft: 0,
  }
}));

// Styled slider label 
const SliderLabel = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  color: theme.palette.mode === 'dark' ? '#f0f0f0' : '#333',
  fontSize: '0.9rem',
  fontWeight: 500,
}));

// Value label for displaying the current slider value
const ValueLabel = styled('span')(({ theme }) => ({
  color: theme.palette.mode === 'dark' ? '#f0c869' : '#0063cc', 
  fontWeight: 600,
  fontSize: '0.95rem',
  marginLeft: '4px',
  minWidth: '20px',
  textAlign: 'center',
}));

// Styled custom slider
const NetworkSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.mode === 'dark' ? '#f0c869' : '#0063cc',
  height: 6,
  padding: '15px 0',
  '& .MuiSlider-thumb': {
    height: 18,
    width: 18,
    backgroundColor: theme.palette.mode === 'dark' ? '#fff' : '#fff',
    boxShadow: theme.palette.mode === 'dark' 
      ? '0 0 8px rgba(240, 200, 105, 0.5)' 
      : '0 0 8px rgba(0, 99, 204, 0.5)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: theme.palette.mode === 'dark' 
        ? '0 0 0 8px rgba(240, 200, 105, 0.3)' 
        : '0 0 0 8px rgba(0, 99, 204, 0.2)',
    },
    '&.Mui-active': {
      boxShadow: theme.palette.mode === 'dark' 
        ? '0 0 0 12px rgba(240, 200, 105, 0.4)' 
        : '0 0 0 12px rgba(0, 99, 204, 0.3)',
    }
  },
  '& .MuiSlider-rail': {
    opacity: 0.4,
    backgroundColor: theme.palette.mode === 'dark' ? '#666' : '#bbb',
  },
  '& .MuiSlider-track': {
    height: 6,
  },
  '& .MuiSlider-mark': {
    backgroundColor: theme.palette.mode === 'dark' ? '#888' : '#bbb',
    height: 8,
    width: 2,
    marginTop: -1,
  },
  '& .MuiSlider-markActive': {
    backgroundColor: theme.palette.mode === 'dark' ? '#fff' : '#666',
  },
}));

// Styled control button
const ControlButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.mode === 'dark' ? '#f0c869' : '#0063cc',
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(240, 200, 105, 0.15)' : 'rgba(0, 99, 204, 0.08)',
  padding: 8,
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' ? 'rgba(240, 200, 105, 0.25)' : 'rgba(0, 99, 204, 0.15)',
    transform: 'translateY(-2px)',
  },
  '&:active': {
    transform: 'translateY(0)',
  }
}));

/**
 * NetworkControls component provides controls for adjusting network visualization parameters
 */
const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onCenterMainNode,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <ControlsContainer>
      <ControlsRow>
        <SliderContainer>
          <SliderLabel>
            Depth
            <ValueLabel>{depth}</ValueLabel>
          </SliderLabel>
          <Tooltip title="Controls how many steps away from the main word to explore. Higher values show more distant connections.">
            <NetworkSlider
              value={depth}
              onChange={(_, value) => onDepthChange(value as number)}
              step={1}
              marks
              min={1}
              max={5}
              aria-label="Network depth"
            />
          </Tooltip>
        </SliderContainer>
        
        <SliderContainer>
          <SliderLabel>
            Breadth
            <ValueLabel>{breadth}</ValueLabel>
          </SliderLabel>
          <Tooltip title="Controls how many related words to show for each node. Higher values show more connections per word.">
            <NetworkSlider
              value={breadth}
              onChange={(_, value) => onBreadthChange(value as number)}
              step={5}
              marks={[
                { value: 5 },
                { value: 15 },
                { value: 25 },
                { value: 35 },
                { value: 50 }
              ]}
              min={5}
              max={50}
              aria-label="Network breadth"
            />
          </Tooltip>
        </SliderContainer>

        <ButtonsToolbar>
          <Tooltip title="Center main word">
            <ControlButton 
              onClick={onCenterMainNode} 
              size="small" 
              aria-label="Center main word"
            >
              <CenterFocusStrongIcon fontSize={isMobile ? "small" : "medium"} />
            </ControlButton>
          </Tooltip>
          
          <Tooltip title="Zoom in">
            <ControlButton 
              onClick={onZoomIn} 
              size="small" 
              aria-label="Zoom in"
            >
              <ZoomInIcon fontSize={isMobile ? "small" : "medium"} />
            </ControlButton>
          </Tooltip>
          
          <Tooltip title="Zoom out">
            <ControlButton 
              onClick={onZoomOut} 
              size="small" 
              aria-label="Zoom out"
            >
              <ZoomOutIcon fontSize={isMobile ? "small" : "medium"} />
            </ControlButton>
          </Tooltip>
          
          <Tooltip title="Reset zoom">
            <ControlButton 
              onClick={onZoomReset} 
              size="small" 
              aria-label="Reset zoom"
            >
              <RestartAltIcon fontSize={isMobile ? "small" : "medium"} />
            </ControlButton>
          </Tooltip>
        </ButtonsToolbar>
      </ControlsRow>
    </ControlsContainer>
  );
};

export default NetworkControls; 