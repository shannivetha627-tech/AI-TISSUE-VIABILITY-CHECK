(function startPatientDashboardPolling() {
  // Live system clock (updates every second)
  const clockEl = document.getElementById('live-clock');
  function updateClock() {
    const now = new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric', month: 'short', day: 'numeric',
      hour: 'numeric', minute: 'numeric', second: 'numeric', hour12: true,
    });
    clockEl.textContent = `Current System Time: ${now} IST`;
  }
  setInterval(updateClock, 1000);
  updateClock();

  const dashboard = document.querySelector("[data-history-url]");
  if (!dashboard) {
    return;
  }

  const historyUrl = dashboard.dataset.historyUrl;
  const latestPanel = document.querySelector("#latest-result-panel");
  const latestLabel = document.querySelector("#latest-prediction-label");
  const latestProbability = document.querySelector("#latest-probability");
  const latestUpdated = document.querySelector("#latest-updated");
  const latestConfidence = document.querySelector("#latest-confidence");
  const historyList = document.querySelector("#history-list");
  const liveStatus = document.querySelector("#live-status");
  const liveStatusText = document.querySelector("#live-status-text");
  let latestHistoryId = Number(
    historyList?.querySelector("[data-history-id]")?.dataset.historyId || 0,
  );
  let polling = false;

  function confidenceFor(probability, prediction) {
    const viable = prediction === "Yes";
    const cert = viable ? probability : (100 - probability);
    if (cert >= 80) {
      return "High";
    }
    if (cert >= 60) {
      return "Moderate";
    }
    return "Low";
  }

  function setStatus(text, connected) {
    liveStatusText.textContent = text;
    liveStatus.classList.toggle("is-error", !connected);
    if (connected) {
      liveStatus.classList.remove("is-polling");
      void liveStatus.offsetWidth;
      liveStatus.classList.add("is-polling");
    }
  }

  function updateLatest(latest) {
    const viable = latest.prediction === "Yes";
    latestPanel.classList.toggle("result-good", viable);
    latestPanel.classList.toggle("result-alert", !viable);
    latestLabel.textContent = viable ? "Viable" : "Not viable";
    latestProbability.innerHTML = `${latest.probability}<small>%</small>`;
    latestUpdated.textContent = `Updated: ${latest.display_time}`;
    latestConfidence.textContent = confidenceFor(latest.probability, latest.prediction);
    latestPanel.classList.remove("live-updated");
    void latestPanel.offsetWidth;
    latestPanel.classList.add("live-updated");
  }

  function makeHistoryItem(item, isLatest) {
    const row = document.createElement("article");
    row.className = `history-row${isLatest ? " history-latest" : ""}`;
    row.dataset.historyId = item.id;

    const dot = document.createElement("span");
    dot.className = "timeline-dot";
    dot.setAttribute("aria-hidden", "true");

    const content = document.createElement("div");
    content.className = "history-main";
    if (isLatest) {
      const badge = document.createElement("span");
      badge.className = "history-badge";
      badge.textContent = "Latest prediction";
      content.appendChild(badge);
    }

    const timestamp = document.createElement("time");
    timestamp.textContent = item.display_time;
    content.appendChild(timestamp);

    const status = document.createElement("strong");
    status.className = item.prediction === "Yes" ? "text-good" : "text-alert";
    status.textContent = item.prediction === "Yes" ? "Viable" : "Not viable";
    content.appendChild(status);

    const probability = document.createElement("span");
    probability.className = "history-confidence";
    probability.textContent = `${item.probability}% confidence`;
    content.appendChild(probability);

    if (item.source && item.source !== "system") {
      const source = document.createElement("small");
      source.textContent = item.source;
      content.appendChild(source);
    }

    row.append(dot, content);
    return row;
  }

  function updateHistory(history) {
    if (!historyList) {
      return;
    }
    historyList.replaceChildren(
      ...history.map((item, index) => makeHistoryItem(item, index === 0)),
    );
    historyList.classList.remove("live-updated");
    void historyList.offsetWidth;
    historyList.classList.add("live-updated");
  }

  async function pollHistory() {
    if (polling) {
      return;
    }
    polling = true;
    try {
      const response = await fetch(`${historyUrl}?page=1&per_page=5`, {
        cache: "no-store",
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error("history request failed");
      }
      const payload = await response.json();
      const latest = payload.latest;
      const history = payload.history || [];
      if (latest) {
        updateLatest(latest);
        updateHistory(history);
        latestHistoryId = latest.id;
      }
      setStatus("Live monitoring connected", true);
    } catch (error) {
      setStatus("Live monitoring temporarily unavailable. Retrying...", false);
    } finally {
      polling = false;
      window.setTimeout(pollHistory, 10000);
    }
  }

  window.setTimeout(pollHistory, 10000);
})();
