const metricsEl = document.querySelector("#metrics");

async function loadMetrics() {
  if (!metricsEl) return;

  try {
    const response = await fetch("reports/metrics.json", { cache: "no-store" });
    if (!response.ok) return;

    const metrics = await response.json();
    const rows = Object.entries(metrics)
      .map(([name, result]) => {
        const label = name
          .split("_")
          .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
          .join(" ");

        return `
          <div class="metric-row">
            <strong>${label}</strong>
            <span>${Math.round(result.accuracy * 100)}% accuracy</span>
          </div>
        `;
      })
      .join("");

    if (rows) {
      metricsEl.className = "";
      metricsEl.innerHTML = rows;
    }
  } catch {
    metricsEl.textContent = "Run the pipeline to generate reports/metrics.json.";
  }
}

loadMetrics();
