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
  background: theme.palette.mode === 'dark' ? 'rgba(30, 30, 30, 0.85)' : 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(8px)',
  borderRadius: '12px',
  padding: '12px 20px',
  boxShadow: theme.palette.mode === 'dark' 
    ? '0 4px 20px rgba(0, 0, 0, 0.5)' 
    : '0 4px 20px rgba(0, 0, 0, 0.15)',
  border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'}`,
  width: '100%',
  maxWidth: '100%',
  margin: '0 auto',
  transition: 'all 0.3s ease',
}));

// Styled component for sliders container
const SlidersContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'row',
  gap: '24px',
  width: '100%',
  '@media (max-width: 600px)': {
    flexDirection: 'column',
    gap: '12px',
  },
});

// Styled component for a single slider
const SliderContainer = styled(Box)({
  flex: 1,
});

// Styled component for buttons container
const ButtonsContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'flex-end',
  gap: '8px',
  marginTop: '8px',
  '@media (max-width: 600px)': {
    marginTop: '12px',
  },
}));

// Styled custom slider
const CustomSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.primary.main,
  height: 6,
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0px 0px 0px 8px ${theme.palette.mode === 'dark' 
        ? 'rgba(90, 90, 255, 0.16)' 
        : 'rgba(0, 0, 255, 0.16)'}`,
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.32,
  },
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

  // Handle depth slider change
  const handleDepthChange = (_: Event, newValue: number | number[]) => {
    onDepthChange(newValue as number);
  };

  // Handle breadth slider change
  const handleBreadthChange = (_: Event, newValue: number | number[]) => {
    onBreadthChange(newValue as number);
  };

  return (
    <ControlsContainer>
      <SlidersContainer>
        <SliderContainer>
          <Typography 
            variant="body2" 
            color="textSecondary" 
            gutterBottom
            sx={{ 
              fontSize: isMobile ? '12px' : '14px',
              fontWeight: 500,
            }}
          >
            Depth: {depth}
          </Typography>
          <CustomSlider
            value={depth}
            onChange={handleDepthChange}
            step={1}
            marks
            min={1}
            max={5}
            aria-label="Network depth"
          />
        </SliderContainer>
        
        <SliderContainer>
          <Typography 
            variant="body2" 
            color="textSecondary" 
            gutterBottom
            sx={{ 
              fontSize: isMobile ? '12px' : '14px',
              fontWeight: 500,
            }}
          >
            Breadth: {breadth}
          </Typography>
          <CustomSlider
            value={breadth}
            onChange={handleBreadthChange}
            step={1}
            marks
            min={1}
            max={10}
            aria-label="Network breadth"
          />
        </SliderContainer>
      </SlidersContainer>
      
      <ButtonsContainer>
        <Tooltip title="Center main word">
          <IconButton 
            onClick={onCenterMainNode} 
            size="small" 
            aria-label="Center main word"
            sx={{ 
              color: theme.palette.text.secondary,
              '&:hover': {
                color: theme.palette.primary.main,
                background: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.08)' 
                  : 'rgba(0, 0, 0, 0.04)',
              },
            }}
          >
            <CenterFocusStrongIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Zoom in">
          <IconButton 
            onClick={onZoomIn} 
            size="small" 
            aria-label="Zoom in"
            sx={{ 
              color: theme.palette.text.secondary,
              '&:hover': {
                color: theme.palette.primary.main,
                background: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.08)' 
                  : 'rgba(0, 0, 0, 0.04)',
              },
            }}
          >
            <ZoomInIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Zoom out">
          <IconButton 
            onClick={onZoomOut} 
            size="small" 
            aria-label="Zoom out"
            sx={{ 
              color: theme.palette.text.secondary,
              '&:hover': {
                color: theme.palette.primary.main,
                background: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.08)' 
                  : 'rgba(0, 0, 0, 0.04)',
              },
            }}
          >
            <ZoomOutIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Reset zoom">
          <IconButton 
            onClick={onZoomReset} 
            size="small" 
            aria-label="Reset zoom"
            sx={{ 
              color: theme.palette.text.secondary,
              '&:hover': {
                color: theme.palette.primary.main,
                background: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.08)' 
                  : 'rgba(0, 0, 0, 0.04)',
              },
            }}
          >
            <RestartAltIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
      </ButtonsContainer>
    </ControlsContainer>
  );
};

export default NetworkControls; 