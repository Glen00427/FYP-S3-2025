import React, { useState, useEffect, useRef } from "react";

export default function NotificationsSettings({
  theme = "light",
  onToggleTheme,
  onClose, // ← NEW (optional)
}) {
  // Traffic alerts
  const [accident, setAccident] = useState(true);
  const [roadwork, setRoadwork] = useState(true);
  const [congestion, setCongestion] = useState(true);
  const [reroute, setReroute] = useState(false);

  // Audio
  const [audioAlerts, setAudioAlerts] = useState(true);
  const [voiceNav, setVoiceNav] = useState(true);
  const [volume, setVolume] = useState(80);
  const [voiceLang, setVoiceLang] = useState("en");

  // App prefs
  const [darkMode, setDarkMode] = useState(theme === "dark");
  const [forecastRange, setForecastRange] = useState("30");
  const [routeType, setRouteType] = useState("fastest");
  const [saveHistory, setSaveHistory] = useState(false);

  // Notification settings
  const [pushNoti, setPushNoti] = useState(true);
  const [smsNoti, setSmsNoti] = useState(false);
  const [speedAlerts, setSpeedAlerts] = useState(false);
  const [alertDistance, setAlertDistance] = useState(2); // km

  function toggleDark() {
    const next = !darkMode;
    setDarkMode(next);
    onToggleTheme?.(next ? "dark" : "light");
  }

  return (
    <div className="ns-page">
      <div className="ns-titlebar" style={{ position: "relative" }}>
        <span className="ns-title-icon">🛠️</span>
        <h1 className="ns-title">Notifications &amp; Settings</h1>
        <p className="ns-subtitle">
          Customise your driving experience and alerts
        </p>

        {/* ✕ close button */}
        {onClose && (
          <button
            aria-label="Close"
            title="Close"
            onClick={onClose}
            className="ns-close"
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              width: 36,
              height: 36,
              borderRadius: 8,
              border: "1px solid var(--border, #e5e7eb)",
              background: "var(--panel, #fff)",
              fontSize: 18,
              lineHeight: "34px",
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        )}
      </div>

      <div className="ns-grid">
        {/* Left column */}
        <section className="ns-col">
          <Card title="Traffic Alert Preferences" icon="🧩">
            <ListItem
              icon="⚠️"
              title="Accident Alerts"
              desc="Notify about accidents on your route"
              right={<Switch on={accident} setOn={setAccident} />}
            />
            <ListItem
              icon="🚧"
              title="Roadwork Alerts"
              desc="Get updates on planned roadworks"
              right={<Switch on={roadwork} setOn={setRoadwork} />}
            />
            <ListItem
              icon="🚗"
              title="Congestion Alerts"
              desc="Alerts about heavy traffic areas"
              right={<Switch on={congestion} setOn={setCongestion} />}
            />
            <ListItem
              icon="🔄"
              title="Auto Re-routing"
              desc="Automatically find alternative routes"
              right={<Switch on={reroute} setOn={setReroute} />}
            />
          </Card>

          <Card title="App Preferences" icon="📝">
            <ListItem
              icon="🌙"
              title="Dark Mode"
              desc="Use dark theme for night driving"
              right={<Switch on={darkMode} setOn={toggleDark} />}
            />

            <Field label="Default Forecast Range">
              <Select
                value={forecastRange}
                onChange={(e) => setForecastRange(e.target.value)}
                options={[
                  { value: "15", label: "15 minutes" },
                  { value: "30", label: "30 minutes (Recommended)" },
                  { value: "45", label: "45 minutes" },
                  { value: "60", label: "60 minutes" },
                ]}
              />
            </Field>

            <Field label="Preferred Route Type">
              <Select
                value={routeType}
                onChange={(e) => setRouteType(e.target.value)}
                options={[
                  { value: "fastest", label: "Fastest Route" },
                  { value: "shortest", label: "Shortest Distance" },
                  { value: "scenic", label: "Scenic Route" },
                  { value: "eco", label: "Eco (Fuel-saving)" },
                ]}
              />
            </Field>

            <ListItem
              icon="📍"
              title="Save Location History"
              desc="Remember frequently visited places"
              right={<Switch on={saveHistory} setOn={setSaveHistory} />}
            />
          </Card>
        </section>

        {/* Right column */}
        <section className="ns-col">
          <Card title="Audio Alert Configuration" icon="🔊">
            <ListItem
              icon="🔔"
              title="Audio Alerts"
              desc="Sound notifications for incidents"
              right={<Switch on={audioAlerts} setOn={setAudioAlerts} />}
            />
            <ListItem
              icon="🎤"
              title="Voice Navigation"
              desc="Spoken turn-by-turn directions"
              right={<Switch on={voiceNav} setOn={setVoiceNav} />}
            />

            <Field label={`Volume Level`} hint={`${volume}`}>
              <input
                className="ns-range"
                type="range"
                min="0"
                max="100"
                value={volume}
                onChange={(e) => setVolume(parseInt(e.target.value, 10))}
              />
              <div className="ns-range-scale">
                <span>Quiet</span>
                <span>Normal</span>
                <span>Loud</span>
              </div>
            </Field>

            <Field label="Voice Language">
              <select
                className="ns-select"
                value={voiceLang}
                onChange={(e) => setVoiceLang(e.target.value)}
              >
                <option value="en">English</option>
                <option value="zh">Chinese (Mandarin)</option>
                <option value="ms">Malay</option>
                <option value="ta">Tamil</option>
              </select>
            </Field>
          </Card>

          <Card title="Notification Settings" icon="🔔">
            <ListItem
              icon="📱"
              title="Push Notifications"
              desc="Receive alerts about traffic and incidents"
              right={<Switch on={pushNoti} setOn={setPushNoti} />}
            />
            <ListItem
              icon="💬"
              title="SMS Alerts"
              desc="Get updates via text messages"
              right={<Switch on={smsNoti} setOn={setSmsNoti} />}
            />
            <ListItem
              icon="⚡"
              title="Speed Limit Alerts"
              desc="Warn when exceeding posted speed"
              right={<Switch on={speedAlerts} setOn={setSpeedAlerts} />}
            />

            <Field label="Alert Distance">
              <input
                className="ns-range"
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={alertDistance}
                onChange={(e) => setAlertDistance(parseFloat(e.target.value))}
              />
              <div className="ns-range-scale">
                <span>100m</span>
                <span>1km ahead</span>
                <span>2km ahead</span>
              </div>
            </Field>
          </Card>
        </section>
      </div>
    </div>
  );
}

/* ---------- tiny building blocks ---------- */

function Card({ title, icon, children }) {
  return (
    <div className="ns-card">
      <div className="ns-card-head">
        <div className="ns-card-title">
          <span className="ns-card-icon">{icon}</span>
          <span>{title}</span>
        </div>
      </div>
      <div className="ns-card-body">{children}</div>
    </div>
  );
}

function ListItem({ icon, title, desc, right }) {
  return (
    <div className="ns-li">
      <div className="ns-li-left">
        <span className="ns-li-icon">{icon}</span>
        <div className="ns-li-text">
          <div className="ns-li-title">{title}</div>
          {desc && <div className="ns-li-desc">{desc}</div>}
        </div>
      </div>
      <div className="ns-li-right">{right}</div>
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <div className="ns-field">
      <div className="ns-field-label">
        <span>{label}</span>
        {hint && <span className="ns-field-hint">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

function Select({ value, onChange, options }) {
  return (
    <select className="ns-select" value={value} onChange={onChange}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function Switch({ on, setOn }) {
  const toggle = () => (typeof setOn === "function" ? setOn(!on) : setOn());
  return (
    <button
      type="button"
      className={`ns-switch ${on ? "is-on" : ""}`}
      role="switch"
      aria-checked={on}
      onClick={toggle}
      aria-label="toggle"
    >
      <span className="ns-switch-knob" />
    </button>
  );
}
