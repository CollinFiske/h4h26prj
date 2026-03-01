import React from "react";

/**
 * Renders either A* evacuation results from the backend or an error message.
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

  const warning = data?.ui?.warning || "ALERT: Evacuate";
  const planStatus = data?.plan?.status || "unknown";
  const chosenCenter = data?.plan?.center || null;
  const pathCells = data?.plan?.path_cells || [];
  const centers = data?.objects?.centers || [];
  const buildings = data?.objects?.buildings_nonsafe || [];
  const fireBlocked = data?.blocked_cells?.fire || [];
  const airBlocked = data?.blocked_cells?.air || [];
  const buildingBlocked = data?.blocked_cells?.buildings || [];

  return (
    <div className="results-panel">
      <h2>Evacuation Plan</h2>
      <pre className="result-pre">
        {[
          `Warning: ${warning}`,
          `Plan status: ${planStatus}`,
          chosenCenter
            ? `Chosen center: ${chosenCenter.name} (${chosenCenter.type})`
            : "Chosen center: none",
          `Path cells: ${pathCells.length}`,
          `Nearby centers: ${centers.length}`,
          `Non-safe buildings: ${buildings.length}`,
          `Blocked cells -> buildings: ${buildingBlocked.length}, fire: ${fireBlocked.length}, air: ${airBlocked.length}`,
          "",
          "Raw response:",
          JSON.stringify(data, null, 2),
        ].join("\n")}
      </pre>
      <h2>Your Evacuation Plan</h2>
      <div className="result-guidance">
        {data.guidance}
      </div>
    </div>
  );
}

export default ResultsDisplay;
