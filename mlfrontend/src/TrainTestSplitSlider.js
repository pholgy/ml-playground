import * as React from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';

function valuetext(value) {
  const testPercent = 100 - value;
  return `Training: ${value}% / Testing: ${testPercent}%`;
}

export default function TrainTestSplitSlider({ value, onChange }) {
  return (
    <Box sx={{ width: 300 }}>
      <Slider
        aria-label="Train Test Split"
        value={value}  // Set value from props
        getAriaValueText={valuetext}
        valueLabelDisplay="auto"
        step={10}
        marks
        min={10}
        max={90}
        onChange={onChange}  // Use onChange from props
      />
    </Box>
  );
}
