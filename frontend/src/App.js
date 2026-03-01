import React, { useState } from "react";
import LocationDisplay from "./components/LocationDisplay";
import EvacuationForm from "./components/EvacuationForm";
import ResultsDisplay from "./components/ResultsDisplay";
import GoogleMapPanel from "./components/GoogleMapPanel";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  /**
   * Sends location + constraints to the A* evacuation backend and stores the response.
   */
  const handleFormSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error("Evacuation request failed:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>CA Fire Detection & Evacuation</h1>
        <p>Real-time fire risk assessment and personalized evacuation routing</p>
      </header>

      {/* Current Location */}
      <LocationDisplay />

      {/* Map (left) + Results (right) side-by-side */}
      <div className="map-results-row">
        <GoogleMapPanel />

        <div className="results-column">
          {loading && <div className="loading">Computing A* evacuation route...</div>}
          {error && <ResultsDisplay error={error} />}
          {prediction && <ResultsDisplay data={prediction} />}
          {!loading && !error && !prediction && (
            <div className="results-panel results-empty">
              <h2>Evacuation Plan</h2>
              <pre className="result-pre">{"Submit the form below to compute an A* route."}</pre>
            </div>
          )}
        </div>
      </div>

      {/* Evacuation Info Form */}
      <EvacuationForm onSubmit={handleFormSubmit} loading={loading} />
    </div>
  );
}

export default App;
