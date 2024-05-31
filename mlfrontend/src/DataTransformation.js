import React, { useState, useEffect } from 'react';

function DataTransformation({ onParamsChange, key }) {
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [scaler, setScaler] = useState('None');

  useEffect(() => {
    // Reset the state when the key changes (dataset changes)
    setNoiseLevel(0);
    setScaler('None');
  }, [key]);

  const handleNoiseChange = (e) => {
    const newNoiseLevel = parseInt(e.target.value, 10); // Parse as an integer
    setNoiseLevel(newNoiseLevel);
    onParamsChange(newNoiseLevel / 100, scaler); // Send the value as a decimal
};

  const handleScalerChange = (e) => {
    const newScaler = e.target.value;
    setScaler(newScaler);
    onParamsChange(noiseLevel, newScaler);
  };

  return (
    <div>
      <div>
      <label>Noise Level: </label>
            <input 
              type="range" 
                min="0" 
                max="100" // Set the max value to 100 (adjust as needed)
                step="1"   // Set the step value to 1 (integer)
                value={noiseLevel} 
                onChange={handleNoiseChange} 
              />
              <span>{noiseLevel}%</span> {/* Display the percentage */}
          </div>
      <div>
        <label>Scaler: </label>
        <select value={scaler} onChange={handleScalerChange}>
          <option value="None">None</option>
          <option value="MinMaxScaler">MinMaxScaler</option>
          <option value="StandardScaler">StandardScaler</option>
        </select>
      </div>
    </div>
  );
}

export default DataTransformation;
