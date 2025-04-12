import React, { useState, useEffect } from 'react';
import Slider from '@mui/material/Slider';
import IconButton from '@mui/material/IconButton';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { styled, useTheme } from '@mui/material/styles';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import NetworkCheckIcon from '@mui/icons-material/NetworkCheck';
import SettingsEthernetIcon from '@mui/icons-material/SettingsEthernet';
import Tooltip from '@mui/material/Tooltip';

const ControlContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: 10,
  left: 10,
  right: 10,
  width: 'calc(100% - 20px)',
  padding: theme.spacing(1, 2),
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(30, 30, 30, 0.85)' 
    : 'rgba(255, 255, 255, 0.9)',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  backdropFilter: 'blur(5px)',
  border: `1px solid ${theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.1)' 
    : 'rgba(0, 0, 0, 0.1)'}`,
  zIndex: 10,
  transition: 'all 0.2s ease',
  '&:hover': {
    boxShadow: theme.shadows[5],
  },
  [theme.breakpoints.down('sm')]: {
    flexDirection: 'column',
    padding: theme.spacing(1),
    gap: theme.spacing(1),
    alignItems: 'stretch',
  }
}));

const ControlSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  flex: 1,
  [theme.breakpoints.down('sm')]: {
    width: '100%',
  }
}));

const ButtonGroup = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(0.5),
  [theme.breakpoints.down('sm')]: {
    justifyContent: 'center',
    marginBottom: theme.spacing(0.5),
  }
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  [theme.breakpoints.down('sm')]: {
    width: '100%',
  }
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.mode === 'dark' ? '#fcbf49' : '#3a86ff',
  height: 4,
  padding: '13px 0',
  '& .MuiSlider-thumb': {
    height: 14,
    width: 14,
    backgroundColor: '#fff',
    boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
    '&:hover': {
      boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    },
    '&:focus, &:hover, &.Mui-active': {
      boxShadow: '0 3px 10px rgba(0,0,0,0.25)',
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.3,
  },
  '& .MuiSlider-mark': {
    backgroundColor: theme.palette.mode === 'dark' ? '#aaa' : '#888',
    height: 8,
    width: 1,
    marginTop: -2,
  },
  '& .MuiSlider-markActive': {
    backgroundColor: theme.palette.mode === 'dark' ? '#fff' : '#000',
  },
}));

const StyledIconButton = styled(IconButton)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.1)' 
    : 'rgba(0, 0, 0, 0.05)',
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' 
      ? 'rgba(255, 255, 255, 0.2)' 
      : 'rgba(0, 0, 0, 0.1)',
  },
  color: theme.palette.mode === 'dark' ? '#fcbf49' : '#3a86ff',
  padding: 6,
}));

const LabelContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  minWidth: 80,
  [theme.breakpoints.down('sm')]: {
    minWidth: 70,
  }
}));

const LabelTypography = styled(Typography)(({ theme }) => ({
  fontSize: '0.85rem',
  fontWeight: 500,
  color: theme.palette.mode === 'dark' ? '#fff' : '#333',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
}));

const ValueTypography = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  color: theme.palette.mode === 'dark' ? '#fcbf49' : '#3a86ff',
  fontWeight: 'bold',
  marginLeft: theme.spacing(1),
  minWidth: 20,
}));

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (newDepth: number) => void;
  onBreadthChange: (newBreadth: number) => void;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onZoomReset?: () => void;
  onCenterMainNode?: () => void;
}

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
  const [localDepth, setLocalDepth] = useState(depth);
  const [localBreadth, setLocalBreadth] = useState(breadth);

  useEffect(() => {
    setLocalDepth(depth);
    setLocalBreadth(breadth);
  }, [depth, breadth]);

  const handleDepthChange = (_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    setLocalDepth(value);
  };

  const handleBreadthChange = (_event: Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    setLocalBreadth(value);
  };

  const handleDepthChangeCommitted = (_event: React.SyntheticEvent | Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    if (value !== depth) {
      onDepthChange(value);
    }
  };

  const handleBreadthChangeCommitted = (_event: React.SyntheticEvent | Event, newValue: number | number[]) => {
    const value = Array.isArray(newValue) ? newValue[0] : newValue;
    if (value !== breadth) {
      onBreadthChange(value);
    }
  };

  return (
    <ControlContainer>
      <ButtonGroup>
        <Tooltip title="Zoom In">
          <StyledIconButton size="small" onClick={onZoomIn} aria-label="Zoom in">
            <ZoomInIcon fontSize="small" />
          </StyledIconButton>
        </Tooltip>
        
        <Tooltip title="Zoom Out">
          <StyledIconButton size="small" onClick={onZoomOut} aria-label="Zoom out">
            <ZoomOutIcon fontSize="small" />
          </StyledIconButton>
        </Tooltip>
        
        <Tooltip title="Reset Zoom">
          <StyledIconButton size="small" onClick={onZoomReset} aria-label="Reset zoom">
            <RestartAltIcon fontSize="small" />
          </StyledIconButton>
        </Tooltip>
        
        <Tooltip title="Center Main Word">
          <StyledIconButton size="small" onClick={onCenterMainNode} aria-label="Center main word">
            <FitScreenIcon fontSize="small" />
          </StyledIconButton>
        </Tooltip>
      </ButtonGroup>
      
      <ControlSection>
        <LabelContainer>
          <LabelTypography>
            <NetworkCheckIcon fontSize="small" sx={{ opacity: 0.7 }} />
            Depth
          </LabelTypography>
          <ValueTypography>{localDepth}</ValueTypography>
        </LabelContainer>
        
        <SliderContainer>
          <Tooltip title="Controls how many steps away from the main word to explore. Higher values show more distant connections.">
            <StyledSlider
              size="small"
              value={localDepth}
              onChange={handleDepthChange}
              onChangeCommitted={handleDepthChangeCommitted}
              min={1}
              max={4}
              step={1}
              marks
              aria-label="Network depth"
            />
          </Tooltip>
        </SliderContainer>
      </ControlSection>
      
      <ControlSection>
        <LabelContainer>
          <LabelTypography>
            <SettingsEthernetIcon fontSize="small" sx={{ opacity: 0.7 }} />
            Breadth
          </LabelTypography>
          <ValueTypography>{localBreadth}</ValueTypography>
        </LabelContainer>
        
        <SliderContainer>
          <Tooltip title="Controls how many related words to show for each node. Higher values show more connections per word.">
            <StyledSlider
              size="small"
              value={localBreadth}
              onChange={handleBreadthChange}
              onChangeCommitted={handleBreadthChangeCommitted}
              min={5}
              max={50}
              step={5}
              marks={[
                { value: 5 },
                { value: 20 },
                { value: 35 },
                { value: 50 }
              ]}
              aria-label="Network breadth"
            />
          </Tooltip>
        </SliderContainer>
      </ControlSection>
    </ControlContainer>
  );
};

export default NetworkControls; 