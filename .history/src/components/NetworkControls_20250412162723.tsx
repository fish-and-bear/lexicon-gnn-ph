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
import Divider from '@mui/material/Divider';

const ControlContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: 0,
  left: '50%',
  transform: 'translateX(-50%)',
  width: 'auto',
  minWidth: 300,
  maxWidth: '90%',
  padding: theme.spacing(2),
  margin: theme.spacing(2),
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(30, 30, 30, 0.9)' 
    : 'rgba(255, 255, 255, 0.9)',
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: theme.shadows[8],
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(2),
  backdropFilter: 'blur(8px)',
  border: `1px solid ${theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.12)' 
    : 'rgba(0, 0, 0, 0.08)'}`,
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: theme.shadows[12],
  },
}));

const ControlSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
}));

const SliderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  flex: 1,
  minWidth: 170,
}));

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: '#fcbf49',
  height: 4,
  '& .MuiSlider-thumb': {
    height: 16,
    width: 16,
    backgroundColor: theme.palette.mode === 'dark' ? '#fff' : '#fcbf49',
    boxShadow: '0 2px 12px rgba(0,0,0,0.2)',
    '&:hover': {
      boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    },
    '&:focus, &:hover, &.Mui-active': {
      boxShadow: '0 3px 12px rgba(0,0,0,0.3)',
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.3,
  },
  '& .MuiSlider-valueLabel': {
    backgroundColor: '#fcbf49',
    fontWeight: 'bold',
  },
}));

const StyledIconButton = styled(IconButton)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'rgba(252, 191, 73, 0.12)' 
    : 'rgba(252, 191, 73, 0.1)',
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' 
      ? 'rgba(252, 191, 73, 0.25)' 
      : 'rgba(252, 191, 73, 0.2)',
  },
  color: theme.palette.mode === 'dark' ? '#fcbf49' : '#d99c00',
  transition: 'all 0.2s',
}));

const LabelTypography = styled(Typography)(({ theme }) => ({
  fontSize: '0.8rem',
  fontWeight: 500,
  color: theme.palette.text.secondary,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
}));

const ValueTypography = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  color: theme.palette.text.secondary,
  marginLeft: theme.spacing(1),
  minWidth: 24,
  textAlign: 'center',
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
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>Network Settings</Typography>
      </Box>
      
      <ControlSection>
        <SliderContainer>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
            <LabelTypography>
              <NetworkCheckIcon fontSize="small" sx={{ opacity: 0.7 }} />
              Depth
            </LabelTypography>
            <ValueTypography>{localDepth}</ValueTypography>
          </Box>
          <StyledSlider
            size="small"
            value={localDepth}
            onChange={handleDepthChange}
            onChangeCommitted={handleDepthChangeCommitted}
            min={1}
            max={5}
            step={1}
            marks
            valueLabelDisplay="auto"
            aria-label="Network depth"
          />
        </SliderContainer>
        
        <SliderContainer>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
            <LabelTypography>
              <SettingsEthernetIcon fontSize="small" sx={{ opacity: 0.7 }} />
              Breadth
            </LabelTypography>
            <ValueTypography>{localBreadth}</ValueTypography>
          </Box>
          <StyledSlider
            size="small"
            value={localBreadth}
            onChange={handleBreadthChange}
            onChangeCommitted={handleBreadthChangeCommitted}
            min={5}
            max={30}
            step={5}
            marks
            valueLabelDisplay="auto"
            aria-label="Network breadth"
          />
        </SliderContainer>
      </ControlSection>
      
      <Divider sx={{ opacity: 0.6, my: 0.5 }} />
      
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
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
      </Box>
    </ControlContainer>
  );
};

export default NetworkControls; 