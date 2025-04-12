import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import { styled } from '@mui/material/styles';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

// Styled components
const ControlContainer = styled('div')(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
  borderRadius: theme.shape.borderRadius,
  marginBottom: theme.spacing(2)
}));

const SliderContainer = styled('div')(({ theme }) => ({
  marginBottom: theme.spacing(3),
  '&:last-child': {
    marginBottom: 0
  }
}));

const LabelContainer = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '8px'
});

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange
}) => {
  const handleDepthChange = useCallback((_event: Event, newValue: number | number[]) => {
    onDepthChange(Array.isArray(newValue) ? newValue[0] : newValue);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_event: Event, newValue: number | number[]) => {
    onBreadthChange(Array.isArray(newValue) ? newValue[0] : newValue);
  }, [onBreadthChange]);

  const depthMarks = [
    { value: 1, label: '1' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4' },
    { value: 5, label: '5' }
  ];

  const breadthMarks = [
    { value: 5, label: '5' },
    { value: 15, label: '15' },
    { value: 25, label: '25' },
    { value: 35, label: '35' },
    { value: 50, label: '50' }
  ];

  return (
    <ControlContainer>
      <Typography variant="h6" gutterBottom>
        Network Settings
      </Typography>
      
      <SliderContainer>
        <LabelContainer>
          <Typography>Network Depth: {depth}</Typography>
          <Tooltip title="Controls how many levels deep the word relationships go. Higher values show more distant connections but may make the graph more complex.">
            <InfoIcon fontSize="small" color="action" />
          </Tooltip>
        </LabelContainer>
        <Slider
          value={depth}
          onChange={handleDepthChange}
          min={1}
          max={5}
          step={1}
          marks={depthMarks}
          valueLabelDisplay="auto"
          aria-label="Network depth"
        />
        <Typography variant="caption" color="textSecondary">
          Shallow (1) ↔ Deep (5)
        </Typography>
      </SliderContainer>

      <SliderContainer>
        <LabelContainer>
          <Typography>Relations per Word: {breadth}</Typography>
          <Tooltip title="Controls how many relationships are shown for each word. Higher values show more connections but may make the graph more crowded.">
            <InfoIcon fontSize="small" color="action" />
          </Tooltip>
        </LabelContainer>
        <Slider
          value={breadth}
          onChange={handleBreadthChange}
          min={5}
          max={50}
          step={5}
          marks={breadthMarks}
          valueLabelDisplay="auto"
          aria-label="Relations per word"
        />
        <Typography variant="caption" color="textSecondary">
          Focused (5) ↔ Expansive (50)
        </Typography>
      </SliderContainer>
    </ControlContainer>
  );
};

export default NetworkControls; 