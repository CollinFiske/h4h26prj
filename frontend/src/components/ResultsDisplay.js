import React from "react";

/**
 * Renders either prediction results from the backend or an error message.
 */
function ResultsDisplay({ data, error }) {
  if (error) {
    return (
      <div className="results-panel">
        <h2>Evacuation Plan</h2>
        <div className="result-error">{error}</div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="results-panel">
      <h2>Your Evacuation Plan</h2>
      <div className="result-guidance">
        {data.guidance}
      </div>
    </div>
  );
}

export default ResultsDisplay;
