import React, { useMemo, useState } from "react";

const demoRoutes = [
  {
    id: "r1",
    from: "239 Compassvale Walk",
    to: "SIM Global Education",
    timeMin: 30,
    distanceKm: 12,
    date: "2025-08-26",
    tag: "Shortest Distance",
    favorite: true,
    recurring: false,
  },
  {
    id: "r2",
    from: "Our Tampines Hub",
    to: "Bugis Junction",
    timeMin: 35,
    distanceKm: 20,
    date: "2025-09-03",
    tag: "Fastest Route",
    favorite: false,
    recurring: false,
  },
  {
    id: "r3",
    from: "SIM Global Education",
    to: "National Stadium",
    timeMin: 25,
    distanceKm: 9,
    date: "2025-08-29",
    tag: "Avoid Highways",
    favorite: false,
    recurring: false,
  },
];

export default function SavedRoutes({
  routes: initialRoutes,
  onStartTrip,
  onNavigate,
  onToggleFavorite,
  onDelete,
  onClose, // ← NEW (optional)
}) {
  const [routes, setRoutes] = useState(initialRoutes || demoRoutes);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all"); // all | fav | rec | recent

  const doToggleFav = (r) => {
    onToggleFavorite?.(r);
    setRoutes((list) =>
      list.map((x) => (x.id === r.id ? { ...x, favorite: !x.favorite } : x))
    );
  };
  const doDelete = (r) => {
    if (!confirm("Delete this saved route?")) return;
    onDelete?.(r);
    setRoutes((list) => list.filter((x) => x.id !== r.id));
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    let arr = routes;
    if (q) {
      arr = arr.filter(
        (r) =>
          r.from.toLowerCase().includes(q) ||
          r.to.toLowerCase().includes(q) ||
          (r.tag || "").toLowerCase().includes(q)
      );
    }
    if (filter === "fav") arr = arr.filter((r) => r.favorite);
    if (filter === "rec") arr = arr.filter((r) => r.recurring);
    if (filter === "recent") {
      const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000;
      arr = arr.filter((r) => new Date(r.date).getTime() >= cutoff);
    }
    return arr;
  }, [routes, query, filter]);

  const stats = useMemo(() => {
    const total = routes.length;
    const favs = routes.filter((r) => r.favorite).length;
    const rec = routes.filter((r) => r.recurring).length;
    const avgDist =
      total > 0
        ? (routes.reduce((s, r) => s + r.distanceKm, 0) / total).toFixed(2)
        : "0.00";
    const avgTime =
      total > 0
        ? Math.round(routes.reduce((s, r) => s + r.timeMin, 0) / total)
        : 0;
    const mostTo =
      routes
        .map((r) => r.to)
        .sort((a, b, _, m = new Map()) => {
          m.set(a, (m.get(a) || 0) + 1);
          m.set(b, (m.get(b) || 0) + 1);
          return (m.get(b) || 0) - (m.get(a) || 0);
        })[0] || "-";
    return { total, favs, rec, avgDist, avgTime, mostTo };
  }, [routes]);

  return (
    <div className="sr-page">
      <div className="sr-header" style={{ position: "relative" }}>
        {/* Close (X) in the top-right */}
        {onClose && (
          <button
            aria-label="Close"
            title="Close"
            onClick={onClose}
            className="sr-close"
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              width: 36,
              height: 36,
              borderRadius: 8,
              border: "1px solid var(--border, #e2e2e2)",
              background: "var(--panel, #fff)",
              fontSize: 18,
              lineHeight: "34px",
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        )}

        <div className="sr-title-wrap">
          <span className="sr-title-icon">⭐</span>
          <h1 className="sr-title">Saved Routes & Favourites</h1>
        </div>
      </div>

      <div className="sr-grid">
        <main className="sr-main">
          <div className="sr-toolbar">
            <div className="sr-search">
              <span className="sr-search-icon">🔎</span>
              <input
                className="sr-search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search routes.."
              />
            </div>
            <div className="sr-filters">
              <button
                className={`sr-chip ${filter === "all" ? "active" : ""}`}
                onClick={() => setFilter("all")}
              >
                All Routes
              </button>
              <button
                className={`sr-chip ${filter === "fav" ? "active" : ""}`}
                onClick={() => setFilter("fav")}
              >
                Favourites
              </button>
              <button
                className={`sr-chip ${filter === "rec" ? "active" : ""}`}
                onClick={() => setFilter("rec")}
              >
                Recurring
              </button>
              <button
                className={`sr-chip ${filter === "recent" ? "active" : ""}`}
                onClick={() => setFilter("recent")}
              >
                Recent
              </button>
            </div>
          </div>

          <div className="sr-list">
            {filtered.map((r) => (
              <article key={r.id} className="sr-route-card">
                <div className="sr-route-head">
                  <div className="sr-route-head-left">
                    <span className="sr-pin">📍</span>
                    <div className="sr-route-title">
                      {r.from} <span className="sr-arrow">→</span> {r.to}
                    </div>
                  </div>
                  <div className="sr-route-head-actions">
                    <button
                      className="sr-icon-btn"
                      title="Toggle favourite"
                      onClick={() => doToggleFav(r)}
                    >
                      {r.favorite ? "⭐" : "☆"}
                    </button>
                    <button
                      className="sr-icon-btn"
                      title="Delete"
                      onClick={() => doDelete(r)}
                    >
                      🗑️
                    </button>
                  </div>
                </div>

                <div className="sr-meta">
                  <span className="sr-meta-item">⏱️ {r.timeMin} min</span>
                  <span className="sr-meta-item">🚗 {r.distanceKm} km</span>
                  <span className="sr-meta-item">
                    📅 {new Date(r.date).toLocaleDateString()}
                  </span>
                  <span
                    className={`sr-tag ${
                      r.tag === "Fastest Route"
                        ? "is-fastest"
                        : r.tag === "Shortest Distance"
                        ? "is-shortest"
                        : "is-other"
                    }`}
                  >
                    {r.tag}
                  </span>
                </div>

                <div className="sr-actions">
                  <button
                    className="sr-btn sr-btn-ghost"
                    onClick={() => onNavigate?.(r)}
                  >
                    ✈️ Navigate
                  </button>
                  <button
                    className="sr-btn sr-btn-primary"
                    onClick={() => onStartTrip?.(r)}
                  >
                    Start Trip
                  </button>
                </div>
              </article>
            ))}

            {filtered.length === 0 && (
              <div className="sr-empty">
                No routes found. Try clearing filters or saving a route.
              </div>
            )}
          </div>
        </main>

        <aside className="sr-aside">
          <div className="sr-card">
            <h3 className="sr-card-title">📈 Route Statistics</h3>
            <div className="sr-kv">
              <span>Total Routes</span>
              <strong>{stats.total}</strong>
            </div>
            <div className="sr-kv">
              <span>Favourites</span>
              <strong>{stats.favs}</strong>
            </div>
            <div className="sr-kv">
              <span>Recurring</span>
              <strong>{stats.rec}</strong>
            </div>
            <div className="sr-kv">
              <span>Avg. Distance</span>
              <strong>{stats.avgDist} km</strong>
            </div>
          </div>

          <div className="sr-card">
            <h3 className="sr-card-title">📊 Travel Analytics</h3>
            <div className="sr-kv-sm">
              <span>Most Visited</span>
              <strong>{stats.mostTo}</strong>
            </div>
            <div className="sr-kv-sm">
              <span>Average Trip Time</span>
              <strong>{stats.avgTime} minutes</strong>
            </div>
            <div className="sr-kv-sm">
              <span>Peak Travel Hours</span>
              <strong>8:00 AM – 9:30 AM</strong>
            </div>
            <div className="sr-kv-sm">
              <span>Total Trips</span>
              <strong>{routes.length} routes planned</strong>
            </div>
          </div>

          <div className="sr-card">
            <h3 className="sr-card-title">🗓️ Recurring Commutes</h3>
            <div className="sr-muted">No recurring commutes set up.</div>
          </div>
        </aside>
      </div>
    </div>
  );
}
