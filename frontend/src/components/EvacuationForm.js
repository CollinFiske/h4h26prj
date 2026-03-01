import React, { useState } from "react";
import Cookies from "js-cookie";

const getInitialState = () => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const isoDate = `${year}-${month}-${day}`;
  const timeValue = now.toTimeString().slice(0, 5);

  const cookieLat = Cookies.get("user_lat");
  const cookieLng = Cookies.get("user_lng");

  return {
    latitude: cookieLat ? String(cookieLat) : "",
    longitude: cookieLng ? String(cookieLng) : "",
    locationName: "",
    date: isoDate,
    time: timeValue,
    hasDisability: false,
    hasPets: false,
    hasKids: false,
    hasMedications: false,
    otherConcerns: "",
  };
};

/**
 * Collects evacuation-relevant information from the user and
 * packages it with their location before calling onSubmit.
 */
function EvacuationForm({ onSubmit, loading }) {
  const [form, setForm] = useState(getInitialState);

  const handleCheckbox = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.checked }));
  };

  const handleText = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const parsedLat = parseFloat(form.latitude);
    const parsedLng = parseFloat(form.longitude);

    const payload = {
      latitude: Number.isFinite(parsedLat) ? parsedLat : null,
      longitude: Number.isFinite(parsedLng) ? parsedLng : null,
      location_name: form.locationName,
      date: form.date,
      time: form.time,
      has_disability: form.hasDisability,
      has_pets: form.hasPets,
      has_kids: form.hasKids,
      has_medications: form.hasMedications,
      other_concerns: form.otherConcerns,
    };

    onSubmit(payload);
  };

  return (
    <form className="evacuation-form" onSubmit={handleSubmit}>
      <h2>Evacuation Details</h2>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="latitude">Latitude</label>
          <input
            id="latitude"
            type="number"
            name="latitude"
            step="any"
            placeholder="e.g., 34.0522"
            value={form.latitude}
            onChange={handleText}
          />
        </div>

        <div className="form-group">
          <label htmlFor="longitude">Longitude</label>
          <input
            id="longitude"
            type="number"
            name="longitude"
            step="any"
            placeholder="e.g., -118.2437"
            value={form.longitude}
            onChange={handleText}
          />
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="locationName">Location name (optional)</label>
        <input
          id="locationName"
          type="text"
          name="locationName"
          placeholder="e.g., Downtown Los Angeles"
          value={form.locationName}
          onChange={handleText}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="date">Date</label>
          <input
            id="date"
            type="date"
            name="date"
            value={form.date}
            onChange={handleText}
          />
        </div>

        <div className="form-group">
          <label htmlFor="time">Time</label>
          <input
            id="time"
            type="time"
            name="time"
            value={form.time}
            onChange={handleText}
          />
        </div>
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="hasDisability"
            checked={form.hasDisability}
            onChange={handleCheckbox}
          />
          I have a disability or mobility limitation
        </label>
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="hasPets"
            checked={form.hasPets}
            onChange={handleCheckbox}
          />
          I have pets that need to be evacuated
        </label>
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="hasKids"
            checked={form.hasKids}
            onChange={handleCheckbox}
          />
          I have children / dependents
        </label>
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            name="hasMedications"
            checked={form.hasMedications}
            onChange={handleCheckbox}
          />
          I require essential medications / medical equipment
        </label>
      </div>

      <div className="form-group">
        <label htmlFor="otherConcerns">Other concerns:</label>
        <textarea
          id="otherConcerns"
          name="otherConcerns"
          placeholder="e.g., large vehicle needed, elderly family member, livestock..."
          value={form.otherConcerns}
          onChange={handleText}
        />
      </div>

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? "Processing..." : "Get Evacuation Plan"}
      </button>
    </form>
  );
}

export default EvacuationForm;
