import React from "react";

/**
 * Renders either prediction results from the backend or an error message.
 */
function ResultsDisplay({ data, error }) {
  if (error) {
    return (
      <div className="results-panel">
        <h2>Prediction Results</h2>
        <pre className="result-pre result-error">{error}</pre>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="results-panel">
      <h2>Prediction Results</h2>
      <pre className="result-pre">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default ResultsDisplay;
