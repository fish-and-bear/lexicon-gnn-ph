import React, { useCallback } from 'react';
import Slider from '@mui/material/Slider';
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
  display: 'inline-flex',
  gap: '6px',
  padding: '0 4px',
  backgroundColor: 'var(--controls-background-color, rgba(255, 255, 255, 0.8))',
  borderRadius: '3px',
  alignItems: 'center',
  height: '20px',
  '@media (max-width: 600px)': {
    width: '100%',
    gap: '4px'
  }
});

const SliderGroup = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  minWidth: '120px',
  height: '100%'
});

const Label = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: '1px',
  color: 'var(--text-color)',
  fontSize: '0.7rem',
  whiteSpace: 'nowrap',
  '& .value': {
    fontWeight: 'bold',
    marginLeft: '2px',
    minWidth: '12px'
  }
});

const CompactSlider = styled(Slider)({
  color: 'var(--primary-color)',
  padding: '0',
  height: 1,
  width: '70px',
  '& .MuiSlider-thumb': {
    height: 6,
    width: 6,
    marginTop: -2,
    '&:hover': {
      boxShadow: '0 0 0 3px var(--primary-color-rgb, rgba(29, 53, 87, 0.16))'
    }
  },
  '& .MuiSlider-track': {
    height: 1
  },
  '& .MuiSlider-rail': {
    height: 1,
    opacity: 0.2
  },
  '& .MuiSlider-mark': {
    display: 'none'
  },
  '& .MuiSlider-markLabel': {
    display: 'none'
  }
});

const StyledInfoIcon = styled(InfoIcon)({
  fontSize: '0.7rem',
  opacity: 0.6,
  cursor: 'help',
  marginTop: '-1px',
  width: '12px',
  height: '12px'
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
          D
          <Tooltip title="Steps away from main word" arrow placement="top">
            <StyledInfoIcon />
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
          B
          <Tooltip title="Relations per word" arrow placement="top">
            <StyledInfoIcon />
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
            { value: 50, label: '50' }
          ]}
          size="small"
        />
      </SliderGroup>
    </ControlsBar>
  );
};

export default NetworkControls; 