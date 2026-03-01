import React from "react";

/**
 * Renders a placeholder instead of backend evacuation payload details.
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
      <h2>Evacuation Plan</h2>
      <pre className="result-pre">Evacuation results placeholder.</pre>
      <h2>Your Evacuation Plan</h2>
      <div className="result-guidance">Detailed route output is hidden.</div>
    </div>
  );
}

export default ResultsDisplay;
