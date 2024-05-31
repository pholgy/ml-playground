import React, { useEffect, useState } from 'react';
import './App.css';
import Modal from './Modal';
import TrainTestSplitSlider from './TrainTestSplitSlider';
import LoadingButton from '@mui/lab/LoadingButton';
import trainingIcon from './training-icon.png';
import logo from './ml_logo.png'; // Adjust the path according to where you've placed the file
import SaveIcon from '@mui/icons-material/Save';
import DataTransformation from './DataTransformation';
import Button from '@mui/material/Button';
function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [isDatasetDropdownOpen, setIsDatasetDropdownOpen] = useState(false);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState('');
  const [trainTestSplit, setTrainTestSplit] = useState(0.2); // Default to 20% for testing
  const [isTraining, setIsTraining] = useState(false); 
  const [isLoading, setIsLoading] = useState(true); // State to control the loading screen
  const [isEvaluatedVisible, setIsEvaluatedVisible] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(0)}%`;
  };  
  const formatRMSE = (value) => {
    return value.toFixed(4);
  };
  const [isSaving, setIsSaving] = useState(false);
  const handleSaveResult = () => {
    setIsSaving(true);
  
    // Implement the save logic here (placeholder)
    console.log("Saving results...");
  
    // Set a 1-second delay before reversing back to the initial state
    setTimeout(() => {
      setIsSaving(false);
    }, 1000);
  };

  const [transformParams, setTransformParams] = useState({
    noiseLevel: 0,
    scaler: 'None',
  });

  const handleTransformParamsChange = (noiseLevel, scaler) => {
    setTransformParams({ noiseLevel, scaler });
  };

  const [showDataTransformation, setShowDataTransformation] = useState(false);
  
  useEffect(() => {
    // Fetch datasets
    console.log('Fetching datasets');
    fetch('http://localhost:8000/api/datasets/')
      .then(response => response.json())
      .then(data => {
        console.log('Datasets received:', data.datasets);
        setDatasets(data.datasets);
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
      });

    // Delay for the loading screen
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500); // 1.5 seconds delay

    return () => clearTimeout(timer); // Cleanup timer
  }, []);

  const [modelParameters, setModelParameters] = useState(null);
  
  useEffect(() => {
    if (selectedDataset) {
      console.log('Fetching models for:', selectedDataset.problem_type);
      fetch(`http://localhost:8000/api/models/${selectedDataset.problem_type}/`)
        .then(response => response.json())
        .then(data => {
          console.log('Models received:', data.models);
          setModels(data.models);
        })
        .catch(error => {
          console.error('Error fetching models:', error);
        });
        
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (selectedDataset) {
      const defaultParams = {
        noiseLevel: selectedDataset.defaultNoiseLevel || 0, 
        scaler: selectedDataset.defaultScaler || 'None',
      };
  
      setTransformParams(defaultParams);
    }
  }, [selectedDataset]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (event.target.closest(".dropdown")) return;
      setIsDatasetDropdownOpen(false);
      setIsModelDropdownOpen(false);
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const toggleDatasetDropdown = () => {
    console.log('Toggling dataset dropdown');
    setIsDatasetDropdownOpen(!isDatasetDropdownOpen);
  };

  const toggleModelDropdown = () => {
    if (!selectedDataset) {
        console.log('No dataset selected, cannot toggle model dropdown');
        setModalContent("Please select a dataset first.");
        setIsModalOpen(true);
        return;
    }
    console.log('Toggling model dropdown');
    setIsModelDropdownOpen(!isModelDropdownOpen);
  };

  const handleDatasetClick = (dataset) => {
    console.log('Dataset clicked:', dataset.name);
    setIsDatasetDropdownOpen(false);

    if (dataset.name === "MNIST DATA") {
      console.log('MNIST DATA selected, showing modal');
      setModalContent("The MNIST dataset is not available at the moment.");
      setIsModalOpen(true);
      return;
    }

    setSelectedDataset(dataset);
    setSelectedModel(null);
    fetch('http://localhost:8000/api/log-dataset/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ datasetName: dataset.name, problemType: dataset.problem_type }),
    })
    .then(response => response.json())
    .then(data => console.log('Dataset logged:', data))
    .catch(error => console.error('Error:', error));
  };

  const handleModelClick = (modelName) => {
    console.log('Model clicked:', modelName);

    if (modelName === "Ridge Regression") {
        console.log('Ridge Regression selected, showing modal');
        setModalContent("The Ridge Regression model is not available at the moment.");
        setIsModalOpen(true);
        return;
    }

    if (!selectedDataset) {
        console.log('No dataset selected, showing modal');
        setModalContent("Please select a dataset first.");
        setIsModalOpen(true);
        return;
    }

    setIsModelDropdownOpen(false);
    setSelectedModel(modelName);

    // Fetch model parameters for the selected model
    fetch(`http://localhost:8000/api/models/${modelName}/parameters/`)
        .then(response => response.json())
        .then(data => {
            console.log('Model parameters received:', data.parameters);
            setModelParameters(data.parameters);
        })
        .catch(error => {
            console.error('Error fetching model parameters:', error);
        });
  };


  const closeModal = () => {
    console.log('Closing modal');
    setIsModalOpen(false);
  };

  const logResults = (results) => {
    fetch('http://localhost:8000/api/log-results/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(results),
    })
    .then(response => response.json())
    .then(data => console.log('Results logged:', data))
    .catch(error => console.error('Error logging results:', error));
  };

  const handleTrainModel = () => {
    if (!selectedDataset || !selectedModel) {
      setModalContent("Please select both a dataset and a model.");
      setIsModalOpen(true);
      setIsSaving(false);
      return;
    }
  
    setIsTraining(true); // Set the button to loading state
    console.log('Training model with dataset:', selectedDataset.name, 'and model:', selectedModel, 'with split:', trainTestSplit);
  
    fetch('http://localhost:8000/api/train-model/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        datasetName: selectedDataset.name,
        modelName: selectedModel,
        splitRatio: 1 - trainTestSplit,
        transformParams: transformParams,
        modelParams: modelParameters, // Include the updated model parameters
      })
    })
      .then(response => response.json())
      .then(data => {
        // Introduce a delay of 3 seconds before processing the response
        setTimeout(() => {
          console.log('Model training response:', data);
          setModalContent("Model trained successfully. Check the console for details.");
          setEvaluationResults(data); // Update this line to store the results, including image URL
          logResults(data); // Assuming logResults is a function you've defined
          setIsModalOpen(true);
          setIsTraining(false); // Reset the loading state after the delay
          setIsEvaluatedVisible(true); // Set the Evaluated message to be visible
        }, 3000); // 3000 milliseconds = 3 seconds
      })
      .catch(error => {
        console.error('Error training model:', error);
        setModalContent("Error training model. Please try again.");
        setIsModalOpen(true);
        setIsTraining(false);
      });
  };  
  
  
  const handleDataTransformationClick = () => {
    if (!selectedDataset || !selectedModel) {
      setModalContent("Please select both a dataset and a model for data transformation.");
      setIsModalOpen(true);
    } else {
      setShowDataTransformation(!showDataTransformation);
    }
  };

  
  if (isLoading) {
    return (
      <div className="full-page-overlay">
        <div className="lds-ellipsis">
          <div></div>
          <div></div>
          <div></div>
          <div></div>
        </div>
      </div>
    );
  }
  

  const handleParameterChange = (paramName, paramValue) => {
    // Parse numerical parameters to integers
    const updatedParamValue = !isNaN(paramValue) ? parseInt(paramValue) : paramValue;
    
    setModelParameters(prevParams => ({
      ...prevParams,
      [paramName]: updatedParamValue
    }));
};

  
  return (
    <div className="app-container">
      <div className="left-side">
        <div className="dropdown-container">
          <div className="dropdown">
            <button onClick={toggleDatasetDropdown} className="dropdown-btn">
              {selectedDataset ? selectedDataset.name : 'Select Dataset'}
            </button>
            {isDatasetDropdownOpen && (
              <ul className="dropdown-content">
                {datasets.map(dataset => (
                  <li key={dataset.name} className="dropdown-item">
                    <button onClick={() => handleDatasetClick(dataset)}>
                      {dataset.name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <br></br>
          <div className="dropdown" style={{ marginTop: '15px' }}>
            <button onClick={toggleModelDropdown} className="dropdown-btn">
              {selectedModel ? selectedModel : 'Select Model'}
            </button>
            {isModelDropdownOpen && (
              <ul className="dropdown-content">
                {models.map(modelName => (
                  <li key={modelName} className="dropdown-item">
                    <button onClick={() => handleModelClick(modelName)}>
                      {modelName}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {selectedModel && modelParameters && (
            <div className="parameters-container">
            <h3>Model Parameters</h3>
            <form>
              {Object.entries(modelParameters).map(([paramName, paramValue]) => (
                <div key={paramName} className="parameter-item">
                  <label htmlFor={paramName}>{paramName}</label>
                  {paramName === 'solver' ? (
                    <select
                      id={paramName}
                      value={paramValue}
                      onChange={(e) => handleParameterChange(paramName, e.target.value)}
                    >
                      {['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'].map(solver => (
                        <option key={solver} value={solver}>{solver}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="range"
                      min={0}
                      max={paramName === 'max_iter' ? 1000 : 10} // Adjust the range based on parameter
                      value={paramValue}
                      onChange={(e) => handleParameterChange(paramName, parseInt(e.target.value))}
                    />
                  )}
                </div>
              ))}
            </form>
          </div>
        )}
          <div>
            <h2>Train Test Split:</h2>
            <TrainTestSplitSlider
              value={trainTestSplit * 100}
              onChange={(e, value) => setTrainTestSplit(value / 100)}
            />
          </div>

        <div>
          <LoadingButton
            onClick={handleTrainModel}
            loading={isTraining}
            variant="contained"
            color="primary"
            startIcon={<img src={trainingIcon} alt="Training" style={{ width: '25px', height: '30px' }} />}
            className="custom-button" // Add this line
          >
            Train Model
          </LoadingButton>
          </div>
          {isEvaluatedVisible && evaluationResults && (
          <div className="evaluation-results">
            {evaluationResults.plot_filename && (
              <div className="plot-image">
                <img src={`${process.env.PUBLIC_URL}/figured.png?timestamp=${new Date().getTime()}`} alt="Model Plot" />
              </div>
            )}
            {evaluationResults.accuracy ? (
              <div>
                <div className="results-row">
                  <span>Accuracy: {formatPercentage(evaluationResults.accuracy)}</span>
                  <span>F1 Score: {formatPercentage(evaluationResults.f1_score)}</span>
                </div>
                <div className="results-row">
                  <span>Recall: {formatPercentage(evaluationResults.recall)}</span>
                  <span>Precision: {formatPercentage(evaluationResults.precision)}</span>
                </div>
              </div>
            ) : (
              <div className="results-row">
                <span>RMSE: {formatRMSE(evaluationResults.rmse)}</span>
                <span>R2: {formatPercentage(evaluationResults.r2_score)}</span>
              </div>
            )}
            <div className={`save-result-container ${
              evaluationResults.accuracy ? 'save-result-button-classification' : 'save-result-button-regression'
            }`}>
              <LoadingButton
                color="secondary"
                onClick={handleSaveResult}
                loading={isSaving}
                loadingPosition="start"
                startIcon={<SaveIcon />}
                variant="contained"
              >
                Save Result
              </LoadingButton>
            </div>
          </div>
        )}


        </div>
      </div>
      <div className="right-side">
        <div className="app-logo">
          <img src={logo} alt="ML Playground Logo" style={{ width: '225px' }} />
        </div>

        <div className="buttons-container" style={{ display: 'flex', flexDirection: 'column', gap: '20px', alignItems: 'flex-start', padding: '20px' }}>
          {/* Data Transformation Button */}
          <div className="button-container">
            <button onClick={handleDataTransformationClick} className="data-transformation-toggle">
              Data Transformation
            </button>
            <div className={`data-transformation-panel ${showDataTransformation ? 'show-data-transformation' : ''}`}>
              <DataTransformation key={selectedDataset ? selectedDataset.name : 'default'} onParamsChange={handleTransformParamsChange} />
            </div>
          </div>

          {/* Learn More Button - Conditionally rendered based on isEvaluatedVisible */}
          {isEvaluatedVisible && (
            <Button
            variant="contained"
            color="primary"
            href="/learnmore-page.html"
            className={`learn-more-button ${isEvaluatedVisible ? 'learn-more-visible' : ''}`} // Use the state to toggle the 'learn-more-visible' class
            style={{
              width: '250px', // Adjust the width as needed
              position: 'absolute', // Absolute positioning to allow top adjustment
              top: '850px', // Adjusted position
              alignSelf: 'flex-start' // Align to the start if your container uses flex
            }}
          >
            Learn More
          </Button>
          )}
        </div>
      </div>

      <Modal isOpen={isModalOpen} onClose={closeModal}>
        <p>{modalContent}</p>
      </Modal>
    </div>
  );
}

export default App;