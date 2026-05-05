const API_BASE = "";

let dashboardState = {
  summary: null,
  forecast7: [],
  forecast30: [],
  districtMap: [],
  crimeTypes: [],
  highRisk: [],
  models: [],
  map: null,
  markersLayer: null,
  selectedCrimeType: "ALL",
  selectedDistrict: "ALL"
};

const aboutText = `
  <p>This dashboard shows Chicago crime forecasting, district risk rankings, crime-type risk, map-based district predictions, and high-risk time periods.</p>
  <p>Risk percentage compares predicted or historical activity against a baseline. It does not predict exact individual incidents.</p>
`;

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function riskClass(level) {
  if (!level) return "low";
  const value = String(level).toLowerCase();

  if (value.includes("very")) return "very-high";
  if (value.includes("high")) return "high";
  if (value.includes("moderate")) return "moderate";

  return "low";
}

async function getJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to fetch ${url}: ${text}`);
  }

  return response.json();
}

function openModal(title, html) {
  document.getElementById("modalTitle").textContent = title;
  document.getElementById("modalBody").innerHTML = html;
  document.getElementById("modalBackdrop").classList.remove("hidden");
}

function closeModal(event) {
  if (event.target.id === "modalBackdrop") {
    forceCloseModal();
  }
}

function forceCloseModal() {
  document.getElementById("modalBackdrop").classList.add("hidden");
}

function refreshDashboard() {
  window.location.reload();
}

function toggleMap() {
  const mapElement = document.getElementById("map");
  const message = document.getElementById("mapCollapsedMessage");
  const button = document.getElementById("toggleMapBtn");

  const isHidden = mapElement.style.display === "none";

  if (isHidden) {
    mapElement.style.display = "block";
    message.classList.add("hidden");
    button.textContent = "Hide Map";

    setTimeout(() => {
      if (dashboardState.map) {
        dashboardState.map.invalidateSize();
      }
    }, 200);
  } else {
    mapElement.style.display = "none";
    message.classList.remove("hidden");
    button.textContent = "Show Map";
  }
}

function renderKPIs() {
  const summary = dashboardState.summary;

  document.getElementById("forecast7Total").textContent =
    formatNumber(summary.forecast_7_day_total);

  document.getElementById("forecast30Total").textContent =
    formatNumber(summary.forecast_30_day_total);

  document.getElementById("topDistrict").textContent =
    `District ${summary.top_district_by_risk.district}`;

  document.getElementById("topDistrictRisk").textContent =
    `${summary.top_district_by_risk.risk_percent}% vs baseline`;

  document.getElementById("bestModel").textContent =
    summary.best_citywide_model;

  document.getElementById("bestModelGroup").textContent =
    summary.best_citywide_model_group;
}

function populateFilters() {
  const crimeSelect = document.getElementById("crimeTypeSelect");
  const districtSelect = document.getElementById("districtSelect");

  crimeSelect.innerHTML = `
    <option value="ALL">All Crime Types</option>
    ${dashboardState.crimeTypes
      .map(row => `<option value="${row.primary_type}">${row.primary_type}</option>`)
      .join("")}
  `;

  districtSelect.innerHTML = `
    <option value="ALL">All Districts</option>
    ${dashboardState.districtMap
      .map(row => `<option value="${row.district}">District ${row.district}</option>`)
      .join("")}
  `;
}

function applyFilters() {
  dashboardState.selectedCrimeType = document.getElementById("crimeTypeSelect").value;
  dashboardState.selectedDistrict = document.getElementById("districtSelect").value;

  renderMap();
  renderDistrictCards();
  renderCrimeTypeCards();
  renderHighRiskTable();
}

function getFilteredDistricts() {
  let rows = [...dashboardState.districtMap];

  if (dashboardState.selectedDistrict !== "ALL") {
    rows = rows.filter(row => String(row.district) === String(dashboardState.selectedDistrict));
  }

  return rows;
}

function getFilteredCrimeTypes() {
  let rows = [...dashboardState.crimeTypes];

  if (dashboardState.selectedCrimeType !== "ALL") {
    rows = rows.filter(row => row.primary_type === dashboardState.selectedCrimeType);
  }

  return rows;
}

function getFilteredHighRisk() {
  let rows = [...dashboardState.highRisk];

  if (dashboardState.selectedDistrict !== "ALL") {
    rows = rows.filter(row => String(row.district) === String(dashboardState.selectedDistrict));
  }

  if (dashboardState.selectedCrimeType !== "ALL") {
    rows = rows.filter(row => row.primary_type === dashboardState.selectedCrimeType);
  }

  return rows;
}

function getCrimeRiskForDistrict(district, crimeType) {
  if (crimeType === "ALL") return null;

  const rows = dashboardState.highRisk
    .filter(row =>
      String(row.district) === String(district) &&
      row.primary_type === crimeType
    )
    .sort((a, b) => Number(b.risk_percent_vs_group_avg) - Number(a.risk_percent_vs_group_avg));

  return rows[0] || null;
}

function renderMap() {
  const rows = getFilteredDistricts();

  if (!window.L) {
    document.getElementById("map").innerHTML = `
      <div style="padding:20px;">
        Map library failed to load. Check internet connection for Leaflet CDN.
      </div>
    `;
    return;
  }

  if (!dashboardState.map) {
    dashboardState.map = L.map("map").setView([41.8781, -87.6298], 10);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(dashboardState.map);

    dashboardState.markersLayer = L.layerGroup().addTo(dashboardState.map);
  }

  dashboardState.markersLayer.clearLayers();

  rows
    .filter(row => row.latitude && row.longitude)
    .forEach(row => {
      const crimeRisk = getCrimeRiskForDistrict(row.district, dashboardState.selectedCrimeType);

      if (dashboardState.selectedCrimeType !== "ALL" && !crimeRisk) {
        return;
      }

      const count = Number(row.predicted_30_day_crime_count || 0);
      const radius = dashboardState.selectedCrimeType === "ALL"
        ? Math.max(7, Math.min(24, count / 48))
        : Math.max(8, Math.min(24, Number(crimeRisk.risk_percent_vs_group_avg || 0) / 25));

      const marker = L.circleMarker([Number(row.latitude), Number(row.longitude)], {
        radius,
        color: "#111",
        fillColor: dashboardState.selectedCrimeType === "ALL" ? "#fff" : "#111",
        fillOpacity: 0.82,
        weight: 2
      });

      const crimeSpecificHTML = crimeRisk
        ? `
          <br><strong>Selected crime pattern</strong><br>
          Crime: ${crimeRisk.primary_type}<br>
          High-risk time: ${crimeRisk.day_name} at ${crimeRisk.hour_label}<br>
          Historical risk: ${crimeRisk.risk_percent_vs_group_avg}%<br>
        `
        : `<br>Selected crime: All Crime Types<br>`;

      marker.bindPopup(`
        <strong>District ${row.district}</strong><br>
        Predicted 30-day district count: ${row.predicted_30_day_crime_count}<br>
        District risk: ${row.display_risk_percent}% ${row.risk_direction}<br>
        District level: ${row.risk_level}
        ${crimeSpecificHTML}
        <br>
        <button onclick="openDistrictDetail('${row.district}')">Open Details</button>
      `);

      marker.addTo(dashboardState.markersLayer);
    });

  setTimeout(() => {
    dashboardState.map.invalidateSize();
  }, 200);
}

function renderForecastList() {
  const container = document.getElementById("forecast7List");

  container.innerHTML = dashboardState.forecast7.map(row => `
    <div class="forecast-row" onclick="openForecastModal('7')">
      <span>${row.date}</span>
      <strong>${row.predicted_crime_count}</strong>
    </div>
  `).join("");
}

function renderDistrictCards() {
  const rows = getFilteredDistricts();

  document.getElementById("districtCards").innerHTML = rows.slice(0, 12).map(row => {
    const crimeRisk = getCrimeRiskForDistrict(row.district, dashboardState.selectedCrimeType);

    const extraLine = crimeRisk
      ? `<small>${crimeRisk.primary_type}: ${crimeRisk.day_name} ${crimeRisk.hour_label}, risk ${crimeRisk.risk_percent_vs_group_avg}%</small><br>`
      : "";

    return `
      <div class="rank-card" onclick="openDistrictDetail('${row.district}')">
        <div class="rank-card-row">
          <strong>District ${row.district}</strong>
          <span class="badge ${riskClass(row.risk_level)}">${row.risk_level}</span>
        </div>
        <small>Predicted 30-day count: ${row.predicted_30_day_crime_count}</small><br>
        <small>Risk: ${row.display_risk_percent}% ${row.risk_direction}</small><br>
        ${extraLine}
      </div>
    `;
  }).join("");
}

function renderCrimeTypeCards() {
  const rows = getFilteredCrimeTypes();

  document.getElementById("crimeTypeCards").innerHTML = rows.slice(0, 12).map(row => `
    <div class="rank-card" onclick="openCrimeTypeDetail('${row.primary_type}')">
      <div class="rank-card-row">
        <strong>${row.primary_type}</strong>
        <span class="badge ${riskClass(row.risk_level)}">${row.risk_level}</span>
      </div>
      <small>Predicted 30-day count: ${row.predicted_30_day_crime_count}</small><br>
      <small>Risk: ${row.display_risk_percent}% ${row.risk_direction}</small><br>
      <small>Confidence: ${row.risk_confidence}</small>
    </div>
  `).join("");

  document.getElementById("crimeSummaryList").innerHTML = rows.slice(0, 7).map(row => `
    <div class="forecast-row" onclick="openCrimeTypeDetail('${row.primary_type}')">
      <span>${row.primary_type}</span>
      <strong>${row.predicted_30_day_crime_count}</strong>
    </div>
  `).join("");
}

function renderHighRiskTable() {
  const rows = getFilteredHighRisk();

  document.getElementById("highRiskTable").innerHTML = rows.slice(0, 12).map(row => `
    <tr onclick="openHighRiskDetail('${row.district}', '${row.primary_type}', '${row.day_name}', '${row.hour_label}', '${row.risk_percent_vs_group_avg}')">
      <td>${row.district}</td>
      <td>${row.primary_type}</td>
      <td>${row.day_name}</td>
      <td>${row.hour_label}</td>
      <td>${row.risk_percent_vs_group_avg}%</td>
    </tr>
  `).join("");
}

function openForecastModal(type) {
  const rows = type === "7" ? dashboardState.forecast7 : dashboardState.forecast30;

  const html = `
    <p>Predicted citywide crime count for the next ${type} days.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Forecast Day</th>
            <th>Predicted Count</th>
            <th>Model</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              <td>${row.date}</td>
              <td>${row.forecast_day}</td>
              <td>${row.predicted_crime_count}</td>
              <td>${row.model || row.selected_model || "Best Model"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal(`${type}-Day Forecast`, html);
}

function openDistrictModal() {
  const html = `
    <p>District ranking based on predicted 30-day crime count and risk versus recent baseline.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>District</th>
            <th>Predicted 30-Day Count</th>
            <th>Expected Count</th>
            <th>Risk %</th>
            <th>Direction</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>
          ${dashboardState.districtMap.map(row => `
            <tr onclick="openDistrictDetail('${row.district}')">
              <td>${row.district}</td>
              <td>${row.predicted_30_day_crime_count}</td>
              <td>${row.expected_30_day_count}</td>
              <td>${row.display_risk_percent}%</td>
              <td>${row.risk_direction}</td>
              <td><span class="badge ${riskClass(row.risk_level)}">${row.risk_level}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal("District Risk Ranking", html);
}

function openCrimeTypeModal() {
  const html = `
    <p>Crime-type risk based on predicted 30-day crime volume compared with recent baseline.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Crime Type</th>
            <th>Predicted 30-Day Count</th>
            <th>Expected Count</th>
            <th>Risk %</th>
            <th>Direction</th>
            <th>Level</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          ${dashboardState.crimeTypes.map(row => `
            <tr onclick="openCrimeTypeDetail('${row.primary_type}')">
              <td>${row.primary_type}</td>
              <td>${row.predicted_30_day_crime_count}</td>
              <td>${row.expected_30_day_count}</td>
              <td>${row.display_risk_percent}%</td>
              <td>${row.risk_direction}</td>
              <td><span class="badge ${riskClass(row.risk_level)}">${row.risk_level}</span></td>
              <td>${row.risk_confidence}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal("Crime Type Risk", html);
}

function openHighRiskModal() {
  const rows = getFilteredHighRisk();

  const html = `
    <p>High-risk periods are based on district, crime type, day of week, and hour.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>District</th>
            <th>Crime Type</th>
            <th>Day</th>
            <th>Hour</th>
            <th>Crime Count</th>
            <th>Share %</th>
            <th>Risk %</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              <td>${row.district}</td>
              <td>${row.primary_type}</td>
              <td>${row.day_name}</td>
              <td>${row.hour_label}</td>
              <td>${row.crime_count}</td>
              <td>${row.share_of_group_percent}%</td>
              <td>${row.risk_percent_vs_group_avg}%</td>
              <td><span class="badge ${riskClass(row.risk_level)}">${row.risk_level}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal("High-Risk Time Periods", html);
}

function openDistrictDetail(district) {
  const row = dashboardState.districtMap.find(item => String(item.district) === String(district));

  if (!row) return;

  let relatedHighRisk = dashboardState.highRisk
    .filter(item => String(item.district) === String(district));

  if (dashboardState.selectedCrimeType !== "ALL") {
    const crimeFiltered = relatedHighRisk.filter(item => item.primary_type === dashboardState.selectedCrimeType);

    if (crimeFiltered.length > 0) {
      relatedHighRisk = crimeFiltered;
    }
  }

  relatedHighRisk = relatedHighRisk.slice(0, 12);

  const html = `
    <div class="modal-grid">
      <div class="detail-card">
        <h4>District ${row.district}</h4>
        <p><strong>Predicted 30-day count:</strong> ${row.predicted_30_day_crime_count}</p>
        <p><strong>Expected count:</strong> ${row.expected_30_day_count}</p>
        <p><strong>Risk:</strong> ${row.display_risk_percent}% ${row.risk_direction}</p>
        <p><strong>Level:</strong> ${row.risk_level}</p>
      </div>

      <div class="detail-card">
        <h4>Map Location</h4>
        <p><strong>Latitude:</strong> ${row.latitude || "N/A"}</p>
        <p><strong>Longitude:</strong> ${row.longitude || "N/A"}</p>
        <p>This marker represents a district-level risk center, not one exact incident.</p>
      </div>
    </div>

    <h3>High-Risk Periods in District ${row.district}</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Crime Type</th>
            <th>Day</th>
            <th>Hour</th>
            <th>Crime Count</th>
            <th>Risk %</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>
          ${relatedHighRisk.map(item => `
            <tr>
              <td>${item.primary_type}</td>
              <td>${item.day_name}</td>
              <td>${item.hour_label}</td>
              <td>${item.crime_count}</td>
              <td>${item.risk_percent_vs_group_avg}%</td>
              <td><span class="badge ${riskClass(item.risk_level)}">${item.risk_level}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal(`District ${district} Prediction Details`, html);
}

function openCrimeTypeDetail(primaryType) {
  const row = dashboardState.crimeTypes.find(item => item.primary_type === primaryType);

  if (!row) return;

  const relatedHighRisk = dashboardState.highRisk
    .filter(item => item.primary_type === primaryType)
    .slice(0, 15);

  const html = `
    <div class="detail-card">
      <h4>${row.primary_type}</h4>
      <p><strong>Predicted 30-day count:</strong> ${row.predicted_30_day_crime_count}</p>
      <p><strong>Expected count:</strong> ${row.expected_30_day_count}</p>
      <p><strong>Risk:</strong> ${row.display_risk_percent}% ${row.risk_direction}</p>
      <p><strong>Level:</strong> ${row.risk_level}</p>
      <p><strong>Confidence:</strong> ${row.risk_confidence}</p>
    </div>

    <h3>High-Risk Time Periods for ${row.primary_type}</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>District</th>
            <th>Day</th>
            <th>Hour</th>
            <th>Crime Count</th>
            <th>Risk %</th>
            <th>Level</th>
          </tr>
        </thead>
        <tbody>
          ${relatedHighRisk.map(item => `
            <tr onclick="openDistrictDetail('${item.district}')">
              <td>${item.district}</td>
              <td>${item.day_name}</td>
              <td>${item.hour_label}</td>
              <td>${item.crime_count}</td>
              <td>${item.risk_percent_vs_group_avg}%</td>
              <td><span class="badge ${riskClass(item.risk_level)}">${item.risk_level}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  openModal(`${primaryType} Prediction Details`, html);
}

function openHighRiskDetail(district, crimeType, day, hour, risk) {
  const html = `
    <div class="detail-card">
      <h4>${crimeType}</h4>
      <p><strong>District:</strong> ${district}</p>
      <p><strong>Day:</strong> ${day}</p>
      <p><strong>Hour:</strong> ${hour}</p>
      <p><strong>Risk percentage:</strong> ${risk}% above/below this group’s normal pattern.</p>
    </div>
  `;

  openModal("High-Risk Period Detail", html);
}

function renderModelComparisonHTML() {
  return `
    <p>Lower MAE and MAPE are better.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Group</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>MAPE</th>
            <th>R2</th>
          </tr>
        </thead>
        <tbody>
          ${dashboardState.models.map(row => `
            <tr>
              <td>${row.model}</td>
              <td>${row.model_group || ""}</td>
              <td>${row.mae}</td>
              <td>${row.rmse}</td>
              <td>${row.mape}%</td>
              <td>${row.r2}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function loadDashboard() {
  try {
    const [
      summary,
      forecast7,
      forecast30,
      districtMap,
      crimeTypes,
      highRisk,
      models
    ] = await Promise.all([
      getJson(`${API_BASE}/api/summary`),
      getJson(`${API_BASE}/api/citywide/forecast?days=7`),
      getJson(`${API_BASE}/api/citywide/forecast?days=30`),
      getJson(`${API_BASE}/api/district/risk-map`),
      getJson(`${API_BASE}/api/crime-types/risk`),
      getJson(`${API_BASE}/api/high-risk-periods?limit=500`),
      getJson(`${API_BASE}/api/model-comparison`)
    ]);

    dashboardState.summary = summary;
    dashboardState.forecast7 = forecast7;
    dashboardState.forecast30 = forecast30;
    dashboardState.districtMap = districtMap;
    dashboardState.crimeTypes = crimeTypes;
    dashboardState.highRisk = highRisk;
    dashboardState.models = models;

    renderKPIs();
    populateFilters();
    renderForecastList();
    renderMap();
    renderDistrictCards();
    renderCrimeTypeCards();
    renderHighRiskTable();

    console.log("Dashboard loaded successfully", dashboardState);
  } catch (error) {
    console.error(error);

    document.body.innerHTML = `
      <div style="padding: 40px; font-family: Arial, sans-serif;">
        <h1>Dashboard failed to load</h1>
        <p>${error.message}</p>
        <p>Test these endpoints directly:</p>
        <ul>
          <li><a href="/api/summary">/api/summary</a></li>
          <li><a href="/api/model-comparison">/api/model-comparison</a></li>
          <li><a href="/api/district/risk-map">/api/district/risk-map</a></li>
          <li><a href="/api/crime-types/risk">/api/crime-types/risk</a></li>
        </ul>
      </div>
    `;
  }
}

loadDashboard();
