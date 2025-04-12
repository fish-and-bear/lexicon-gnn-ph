import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import InfoIcon from '@mui/icons-material/Info';
import Box from '@mui/material/Box';
import { styled } from '@mui/material/styles';

interface NetworkControlsProps {
  depth: number;
  breadth: number;
  onDepthChange: (depth: number) => void;
  onBreadthChange: (breadth: number) => void;
}

const ControlsBar = styled(Box)({
  display: 'flex',
  gap: '12px',
  padding: '4px 8px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.8))',
  borderRadius: '4px',
  alignItems: 'center',
  minWidth: 'fit-content',
  maxWidth: '600px',
  '@media (max-width: 600px)': {
    flexDirection: 'column',
    width: '100%',
    gap: '8px'
  }
});

const SliderGroup = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  flex: 1,
  minWidth: '180px',
  '@media (max-width: 600px)': {
    width: '100%'
  }
});

const Label = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  color: 'var(--text-color)',
  fontSize: '0.75rem',
  whiteSpace: 'nowrap',
  minWidth: '40px',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '4px',
    minWidth: '20px'
  }
});

const CompactSlider = styled(Slider)({
  color: 'var(--primary-color)',
  height: 2,
  padding: '8px 0',
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    '&:hover': {
      boxShadow: '0 0 0 4px var(--primary-color-rgb, rgba(29, 53, 87, 0.16))'
    }
  },
  '& .MuiSlider-track': {
    height: 2
  },
  '& .MuiSlider-rail': {
    height: 2,
    opacity: 0.2
  },
  '& .MuiSlider-mark': {
    height: 4,
    width: 1,
    opacity: 0.3
  }
});

const NetworkControls: React.FC<NetworkControlsProps> = ({
  depth,
  breadth,
  onDepthChange,
  onBreadthChange
}) => {
  const handleDepthChange = useCallback((_: Event, value: number | number[]) => {
    onDepthChange(Array.isArray(value) ? value[0] : value);
  }, [onDepthChange]);

  const handleBreadthChange = useCallback((_: Event, value: number | number[]) => {
    onBreadthChange(Array.isArray(value) ? value[0] : value);
  }, [onBreadthChange]);

  return (
    <ControlsBar>
      <SliderGroup>
        <Label>
          Depth
          <Tooltip title="Steps away from main word" arrow placement="top">
            <InfoIcon sx={{ fontSize: '0.875rem', opacity: 0.7, cursor: 'help' }} />
          </Tooltip>
          <span className="value">{depth}</span>
        </Label>
        <CompactSlider
          value={depth}
          onChange={handleDepthChange}
          min={1}
          max={4}
          step={1}
          marks
          size="small"
        />
      </SliderGroup>

      <SliderGroup>
        <Label>
          Breadth
          <Tooltip title="Relations per word" arrow placement="top">
            <InfoIcon sx={{ fontSize: '0.875rem', opacity: 0.7, cursor: 'help' }} />
          </Tooltip>
          <span className="value">{breadth}</span>
        </Label>
        <CompactSlider
          value={breadth}
          onChange={handleBreadthChange}
          min={5}
          max={50}
          step={5}
          marks={[
            { value: 5, label: '5' },
            { value: 20 },
            { value: 35 },
            { value: 50 }
          ]}
          size="small"
        />
      </SliderGroup>
    </ControlsBar>
  );
};

export default NetworkControls; 