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
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  gap: '20px',
  '@media (max-width: 600px)': {
    padding: '10px 14px',
    flexDirection: 'column',
    alignItems: 'stretch',
  }
}));

// Styled component for slider section
const SliderSection = styled(Box)({
  flex: 1,
  minWidth: '160px',
});

// Styled heading for slider
const SliderHeading = styled(Typography)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  marginBottom: '6px',
  color: theme.palette.text.primary,
  fontSize: '0.9rem',
  fontWeight: 500,
}));

// Styled slider value
const SliderValue = styled('span')(({ theme }) => ({
  color: theme.palette.primary.main,
  fontWeight: 600,
  fontSize: '0.95rem',
  minWidth: '20px',
  textAlign: 'center',
}));

// Styled custom slider
const NetworkSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.primary.main,
  height: 6,
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: theme.palette.background.paper,
    boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0 0 0 8px ${theme.palette.primary.main}20`,
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.4,
  },
  '& .MuiSlider-mark': {
    backgroundColor: theme.palette.text.secondary,
    height: 8,
    width: 2,
    marginTop: -1,
  },
  '& .MuiSlider-markActive': {
    backgroundColor: theme.palette.background.paper,
  },
}));

// Styled buttons container
const ButtonGroup = styled(Box)({
  display: 'flex',
  gap: '8px',
  marginLeft: 'auto',
  '@media (max-width: 600px)': {
    marginLeft: 0,
    justifyContent: 'center',
  }
});

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
      <SliderSection>
        <Tooltip title="How many steps away from the main word to explore" placement="top">
          <SliderHeading>
            Depth
            <SliderValue>{depth}</SliderValue>
          </SliderHeading>
        </Tooltip>
        <NetworkSlider
          value={depth}
          onChange={(_, value) => onDepthChange(value as number)}
          step={1}
          marks
          min={1}
          max={5}
          aria-label="Network depth"
        />
      </SliderSection>
      
      <SliderSection>
        <Tooltip title="How many connections to show per word" placement="top">
          <SliderHeading>
            Breadth
            <SliderValue>{breadth}</SliderValue>
          </SliderHeading>
        </Tooltip>
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
      </SliderSection>

      <ButtonGroup>
        <Tooltip title="Center main word">
          <IconButton 
            onClick={onCenterMainNode}
            size="small"
            color="primary"
          >
            <CenterFocusStrongIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Zoom in">
          <IconButton 
            onClick={onZoomIn}
            size="small" 
            color="primary"
          >
            <ZoomInIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Zoom out">
          <IconButton 
            onClick={onZoomOut}
            size="small"
            color="primary" 
          >
            <ZoomOutIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Reset zoom">
          <IconButton 
            onClick={onZoomReset}
            size="small"
            color="primary"
          >
            <RestartAltIcon fontSize={isMobile ? "small" : "medium"} />
          </IconButton>
        </Tooltip>
      </ButtonGroup>
    </ControlsContainer>
  );
};

export default NetworkControls; 