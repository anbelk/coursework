const {DeckGL, OrthographicView, ScatterplotLayer, TextLayer} = deck;

const deckEl = document.getElementById("deck");
const paperPanel = document.getElementById("paperPanel");
const paperPanelContent = document.getElementById("paperPanelContent");
const closePaperPanel = document.getElementById("closePaperPanel");
const clusterTooltip = document.getElementById("clusterTooltip");
const authorInput = document.getElementById("authorInput");
const suggestionsEl = document.getElementById("suggestions");
const modal = document.getElementById("modal");
const modalContent = document.getElementById("modalContent");
const closeModal = document.getElementById("closeModal");

let deckgl;
let mapState = {papers: [], fine: [], meta: []};
let viewState = {target: [0, 0, 0], zoom: 0};
const collisionExt = deck.CollisionFilterExtension ? [new deck.CollisionFilterExtension()] : [];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markdownLinks(text) {
  const safe = escapeHtml(text);
  return safe.replace(/\[([^\]]+)\]\((https:\/\/openalex\.org\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function truncate(value, maxChars = 90) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= maxChars) return text;
  const slice = text.slice(0, maxChars - 1);
  const lastSpace = slice.lastIndexOf(" ");
  return `${slice.slice(0, lastSpace > 40 ? lastSpace : maxChars - 1).trim()}…`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function fitView(points) {
  if (!points.length) return {target: [0, 0, 0], zoom: 0};
  const xs = points.map((d) => d.x);
  const ys = points.map((d) => d.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1);
  return {
    target: [(minX + maxX) / 2, (minY + maxY) / 2, 0],
    zoom: Math.log2(Math.max(deckEl.clientWidth, deckEl.clientHeight) / span) - 1.2,
  };
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function fadeIn(zoom, lo, hi) {
  return clamp((zoom - lo) / Math.max(0.001, hi - lo), 0, 1);
}

function fadeOut(zoom, lo, hi) {
  return 1 - fadeIn(zoom, lo, hi);
}

function roundedCount(value) {
  const step = value >= 1000 ? 100 : value >= 100 ? 50 : 10;
  return Math.round(value / step) * step;
}

function worldBounds() {
  const scale = 2 ** viewState.zoom;
  const halfW = deckEl.clientWidth / scale / 2;
  const halfH = deckEl.clientHeight / scale / 2;
  const [cx, cy] = viewState.target;
  return {minX: cx - halfW, maxX: cx + halfW, minY: cy - halfH, maxY: cy + halfH};
}

function visiblePaperTitles() {
  if (viewState.zoom < 5.4) return [];
  const b = worldBounds();
  const visible = mapState.papers.filter((p) => p.x >= b.minX && p.x <= b.maxX && p.y >= b.minY && p.y <= b.maxY);
  return visible.slice(0, 140);
}

function clusterLabelOpacity(type) {
  if (type === "meta") return fadeOut(viewState.zoom, 1.9, 3.0);
  if (type === "fine") return Math.min(fadeIn(viewState.zoom, 2.3, 3.3), fadeOut(viewState.zoom, 5.0, 5.8));
  return fadeIn(viewState.zoom, 5.3, 6.1);
}

function renderLayers() {
  const metaOpacity = clusterLabelOpacity("meta");
  const fineOpacity = clusterLabelOpacity("fine");
  const paperTitleOpacity = clusterLabelOpacity("paper");
  const titleData = visiblePaperTitles();

  const paperLayer = new ScatterplotLayer({
    id: "papers",
    data: mapState.papers,
    pickable: true,
    radiusUnits: "pixels",
    getRadius: 1.7,
    getFillColor: (d) => [...d.color, 150],
    getPosition: (d) => [d.x, d.y],
    onClick: ({object}) => object && showPaper(object.paper_id),
  });

  const metaLabels = new TextLayer({
    id: "meta_labels",
    data: mapState.meta,
    pickable: true,
    opacity: metaOpacity,
    extensions: collisionExt,
    collisionEnabled: true,
    getCollisionPriority: (d) => d.paper_count,
    getPosition: (d) => [d.x, d.y],
    getText: (d) => `${d.label}\n~${roundedCount(d.paper_count)}`,
    getSize: (d) => Math.min(34, 16 + 3 * Math.log2(Math.max(2, d.paper_count))),
    getTextAnchor: "middle",
    getAlignmentBaseline: "center",
    getColor: [15, 23, 42, 235],
    background: true,
    getBackgroundColor: [255, 255, 255, 218],
    backgroundPadding: [8, 5],
    onClick: ({object}) => object && showCluster(`meta_${object.id}`),
  });

  const fineLabels = new TextLayer({
    id: "fine_labels",
    data: mapState.fine.filter((d) => d.paper_count >= 8),
    pickable: true,
    opacity: fineOpacity,
    extensions: collisionExt,
    collisionEnabled: true,
    getCollisionPriority: (d) => d.paper_count,
    getPosition: (d) => [d.x, d.y],
    getText: (d) => d.label,
    getSize: (d) => Math.min(19, 10 + 1.5 * Math.log2(Math.max(2, d.paper_count))),
    getTextAnchor: "middle",
    getAlignmentBaseline: "center",
    getColor: [15, 23, 42, 230],
    background: true,
    getBackgroundColor: [255, 255, 255, 200],
    backgroundPadding: [5, 3],
    onClick: ({object}) => object && showCluster(`fine_${object.id}`),
  });

  const paperTitles = new TextLayer({
    id: "paper_titles",
    data: titleData,
    pickable: true,
    opacity: paperTitleOpacity,
    extensions: collisionExt,
    collisionEnabled: true,
    getCollisionPriority: () => 1,
    getPosition: (d) => [d.x, d.y],
    getText: (d) => d.title_short,
    getSize: 10,
    getTextAnchor: "middle",
    getAlignmentBaseline: "bottom",
    getColor: [15, 23, 42, 225],
    background: true,
    getBackgroundColor: [255, 255, 255, 210],
    backgroundPadding: [3, 2],
    onClick: ({object}) => object && showPaper(object.paper_id),
  });

  return [paperLayer, metaLabels, fineLabels, paperTitles];
}

function render() {
  const layers = renderLayers();
  if (!deckgl) {
    deckgl = new DeckGL({
      container: deckEl,
      views: new OrthographicView({controller: true}),
      initialViewState: viewState,
      controller: true,
      layers,
      onViewStateChange: ({viewState: next}) => {
        viewState = next;
        deckgl.setProps({layers: renderLayers()});
      },
    });
  } else {
    deckgl.setProps({viewState, layers});
  }
}

async function loadMap() {
  mapState = await fetchJson("/api/map/state");
  viewState = fitView(mapState.papers);
  render();
}

async function showPaper(paperId) {
  const paper = await fetchJson(`/api/papers/${paperId}`);
  const authors = (paper.authors || []).map((a) => escapeHtml(a.name || a.author_id || "")).filter(Boolean).join(", ");
  paperPanelContent.innerHTML = `
    <h2>${escapeHtml(paper.title)}</h2>
    <div class="meta-line">${escapeHtml(paper.year || "")} · ${authors}</div>
    <p>${escapeHtml(paper.abstract)}</p>
    <dl>
      <dt>Fine topic</dt><dd>${escapeHtml(paper.fine_label)}</dd>
      <dt>Meta-cluster</dt><dd>${escapeHtml(paper.meta_label)}</dd>
    </dl>
    <a href="${paper.openalex_url}" target="_blank" rel="noreferrer">Open in OpenAlex</a>
  `;
  paperPanel.classList.remove("hidden");
}

async function showCluster(nodeId) {
  const info = await fetchJson(`/api/clusters/info/${nodeId}`);
  const terms = (info.top_terms || []).map((t) => escapeHtml(t.term || "")).join(", ");
  const reps = (info.representative_papers || []).slice(0, 3).map((p) => `<li>${escapeHtml(p.title || "")}</li>`).join("");
  clusterTooltip.innerHTML = `
    <h3>${escapeHtml(info.label)}</h3>
    <p class="muted">${info.paper_count} papers</p>
    <p><strong>Terms:</strong> ${terms}</p>
    <ul>${reps}</ul>
  `;
  clusterTooltip.classList.remove("hidden");
  clearTimeout(showCluster.timer);
  showCluster.timer = setTimeout(() => clusterTooltip.classList.add("hidden"), 7000);
}

closePaperPanel.addEventListener("click", () => paperPanel.classList.add("hidden"));

let searchTimer = null;
authorInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(searchAuthors, 200);
});

async function searchAuthors() {
  const q = authorInput.value.trim();
  if (q.length < 2) {
    suggestionsEl.innerHTML = "";
    return;
  }
  suggestionsEl.innerHTML = `<div class="muted suggestion-note">Searching...</div>`;
  try {
    const authors = await fetchJson(`/api/authors/search?q=${encodeURIComponent(q)}&limit=10`);
    if (!authors.length) {
      suggestionsEl.innerHTML = `<div class="muted suggestion-note">No authors found</div>`;
      return;
    }
    suggestionsEl.innerHTML = authors.map((author) => {
      const papers = (author.last_papers || []).slice(-2).map((p) => escapeHtml(p.title || "")).join(" · ");
      return `
        <div class="suggestion" data-author-id="${author.author_id}" data-name="${escapeHtml(author.display_name)}">
          ${escapeHtml(author.display_name)}
          <small>${author.n_papers} papers · ${escapeHtml(author.author_id)}</small>
          <span>${papers}</span>
        </div>
      `;
    }).join("");
  } catch (error) {
    suggestionsEl.innerHTML = `<div class="muted suggestion-note">Search failed</div>`;
  }
}

suggestionsEl.addEventListener("click", (event) => {
  const item = event.target.closest(".suggestion");
  if (!item) return;
  suggestionsEl.innerHTML = "";
  authorInput.value = item.dataset.name || "";
  runRecommendations({author_id: item.dataset.authorId, name: item.dataset.name});
});

async function runRecommendations(author) {
  modal.classList.remove("hidden");
  modalContent.innerHTML = `<h2>Recommendations for ${escapeHtml(author.name)}</h2><p>Running LLM reranking pipeline...</p>`;
  try {
    const result = await fetchJson(`/api/recommendations/${author.author_id}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({top_n: 10, top_k: 3, workers: 8}),
    });
    renderRecommendations(result);
  } catch (error) {
    modalContent.innerHTML = `<h2>Recommendation error</h2><p>${escapeHtml(error.message)}</p>`;
  }
}

function renderRecommendations(result) {
  if (!result.selected || !result.selected.length) {
    modalContent.innerHTML = `
      <h2>Recommendations for ${escapeHtml(result.user_name || result.user_id)}</h2>
      <p>${escapeHtml(result.message || "Достаточно релевантные потенциальные соавторы не найдены.")}</p>
    `;
    return;
  }
  const evidenceById = Object.fromEntries((result.evidence || []).map((ev) => [ev.candidate_id, ev]));
  const cards = result.selected.map((cand) => {
    const ev = evidenceById[cand.author_id] || {summary: "", links: []};
    const links = (ev.links || []).map((link) => {
      const userT = truncate(link.user_paper_title || "user paper", 54);
      const candT = truncate(link.candidate_paper_title || "candidate paper", 54);
      return `
        <li class="evidence-link">
          <a href="${link.user_url}" target="_blank" rel="noreferrer">${escapeHtml(userT)}</a>
          <span>↔</span>
          <a href="${link.candidate_url}" target="_blank" rel="noreferrer">${escapeHtml(candT)}</a>
          <div class="link-reason"><strong>Reason:</strong> ${escapeHtml(link.reason)}</div>
        </li>
      `;
    }).join("");
    return `
      <div class="rec-card">
        <h3><a href="${cand.author_url}" target="_blank" rel="noreferrer">${escapeHtml(cand.name)}</a></h3>
        <div class="rec-type">Recommendation type: ${escapeHtml(ev.recommendation_type || cand.recommendation_type || "direct topical match")}</div>
        <p>${escapeHtml(ev.summary || "")}</p>
        <ul>${links}</ul>
      </div>
    `;
  }).join("");
  modalContent.innerHTML = `
    <h2>Recommendations for ${escapeHtml(result.user_name || result.user_id)}</h2>
    <p>${markdownLinks(result.overview || result.message || "")}</p>
    ${cards}
  `;
}

function closeRecommendationModal() {
  modal.classList.add("hidden");
}

closeModal.addEventListener("click", closeRecommendationModal);
modal.addEventListener("click", (event) => {
  if (event.target === modal) closeRecommendationModal();
});

loadMap().catch((error) => {
  deckEl.innerHTML = `<div class="load-error">Failed to load map: ${escapeHtml(error.message)}</div>`;
});
