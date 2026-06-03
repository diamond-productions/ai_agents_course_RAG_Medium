const PAGE_SIZE = 12;

const state = {
  articles: [],
  filtered: [],
  selectedIndex: -1,
  page: 1,
};

const els = {
  datasetStatus: document.querySelector("#datasetStatus"),
  datasetSelect: document.querySelector("#datasetSelect"),
  fileInput: document.querySelector("#fileInput"),
  searchInput: document.querySelector("#searchInput"),
  tagFilter: document.querySelector("#tagFilter"),
  yearFilter: document.querySelector("#yearFilter"),
  sortSelect: document.querySelector("#sortSelect"),
  totalCount: document.querySelector("#totalCount"),
  visibleCount: document.querySelector("#visibleCount"),
  tagCount: document.querySelector("#tagCount"),
  authorCount: document.querySelector("#authorCount"),
  tagChart: document.querySelector("#tagChart"),
  chartCaption: document.querySelector("#chartCaption"),
  articleList: document.querySelector("#articleList"),
  articleTemplate: document.querySelector("#articleTemplate"),
  resultRange: document.querySelector("#resultRange"),
  pageLabel: document.querySelector("#pageLabel"),
  prevPage: document.querySelector("#prevPage"),
  nextPage: document.querySelector("#nextPage"),
  detailPanel: document.querySelector("#detailPanel"),
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell);
  if (row.some((value) => value.length > 0)) rows.push(row);

  const headers = rows.shift()?.map((header) => header.trim()) ?? [];
  return rows.map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = values[index] ?? "";
    });
    return normalizeArticle(record);
  });
}

function normalizeArticle(record) {
  const timestamp = new Date(record.timestamp);
  return {
    title: record.title || "Untitled",
    text: record.text || "",
    url: record.url || "",
    authors: parseList(record.authors),
    timestamp: Number.isNaN(timestamp.getTime()) ? null : timestamp,
    tags: parseList(record.tags),
  };
}

function parseList(value) {
  if (!value) return [];
  try {
    return JSON.parse(value.replaceAll("'", '"')).filter(Boolean);
  } catch {
    return value
      .replace(/^\[|\]$/g, "")
      .split(",")
      .map((item) => item.trim().replace(/^['"]|['"]$/g, ""))
      .filter(Boolean);
  }
}

async function loadDataset(path) {
  els.datasetStatus.textContent = "Loading dataset...";
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Unable to load ${path}. Start a server from the project root.`);
  }
  const csv = await response.text();
  setArticles(parseCsv(csv), path.split("/").pop());
}

function setArticles(articles, label) {
  state.articles = articles;
  state.selectedIndex = articles.length ? 0 : -1;
  state.page = 1;
  els.datasetStatus.textContent = `${articles.length.toLocaleString()} rows from ${label}`;
  hydrateFilters();
  applyFilters();
}

function hydrateFilters() {
  const selectedTag = els.tagFilter.value;
  const selectedYear = els.yearFilter.value;
  const tags = countBy(state.articles.flatMap((article) => article.tags));
  const years = [...new Set(state.articles.map((article) => getYear(article)).filter(Boolean))].sort(
    (a, b) => b - a,
  );

  els.tagFilter.innerHTML = '<option value="">All tags</option>';
  [...tags.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .forEach(([tag, count]) => {
      const option = new Option(`${tag} (${count})`, tag);
      els.tagFilter.add(option);
    });
  els.tagFilter.value = selectedTag;

  els.yearFilter.innerHTML = '<option value="">All</option>';
  years.forEach((year) => els.yearFilter.add(new Option(year, year)));
  els.yearFilter.value = selectedYear;
}

function applyFilters() {
  const query = els.searchInput.value.trim().toLowerCase();
  const tag = els.tagFilter.value;
  const year = els.yearFilter.value;
  const sort = els.sortSelect.value;

  state.filtered = state.articles.filter((article) => {
    const matchesQuery =
      !query ||
      [article.title, article.text, article.url, article.authors.join(" "), article.tags.join(" ")]
        .join(" ")
        .toLowerCase()
        .includes(query);
    const matchesTag = !tag || article.tags.includes(tag);
    const matchesYear = !year || String(getYear(article)) === year;
    return matchesQuery && matchesTag && matchesYear;
  });

  state.filtered.sort((a, b) => {
    if (sort === "oldest") return dateValue(a) - dateValue(b);
    if (sort === "title") return a.title.localeCompare(b.title);
    if (sort === "longest") return b.text.length - a.text.length;
    return dateValue(b) - dateValue(a);
  });

  const maxPage = Math.max(1, Math.ceil(state.filtered.length / PAGE_SIZE));
  state.page = Math.min(state.page, maxPage);
  if (!state.filtered.includes(state.articles[state.selectedIndex])) {
    state.selectedIndex = state.articles.indexOf(state.filtered[0]);
  }

  renderStats();
  renderChart();
  renderArticles();
  renderDetail();
}

function renderStats() {
  els.totalCount.textContent = state.articles.length.toLocaleString();
  els.visibleCount.textContent = state.filtered.length.toLocaleString();
  els.tagCount.textContent = countBy(state.articles.flatMap((article) => article.tags)).size.toLocaleString();
  els.authorCount.textContent = countBy(state.articles.flatMap((article) => article.authors)).size.toLocaleString();
}

function renderChart() {
  const tags = [...countBy(state.filtered.flatMap((article) => article.tags)).entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  const max = Math.max(...tags.map(([, count]) => count), 1);
  els.chartCaption.textContent = tags.length ? `${tags.length} tags in current view` : "No matches";
  els.tagChart.innerHTML = tags
    .map(
      ([tag, count]) => `
        <div class="bar-row">
          <span class="bar-label" title="${escapeHtml(tag)}">${escapeHtml(tag)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${(count / max) * 100}%"></span></span>
          <span class="bar-value">${count}</span>
        </div>
      `,
    )
    .join("");
}

function renderArticles() {
  els.articleList.innerHTML = "";
  const start = (state.page - 1) * PAGE_SIZE;
  const pageItems = state.filtered.slice(start, start + PAGE_SIZE);
  pageItems.forEach((article) => {
    const row = els.articleTemplate.content.firstElementChild.cloneNode(true);
    row.classList.toggle("is-active", state.articles[state.selectedIndex] === article);
    row.querySelector(".article-title").textContent = article.title;
    row.querySelector(".article-meta").textContent = `${formatDate(article.timestamp)} | ${
      article.authors.join(", ") || "Unknown author"
    } | ${article.text.length.toLocaleString()} chars`;
    row.querySelector(".article-tags").textContent = article.tags.join(", ") || "No tags";
    row.addEventListener("click", () => {
      state.selectedIndex = state.articles.indexOf(article);
      renderArticles();
      renderDetail();
    });
    els.articleList.append(row);
  });

  const end = Math.min(start + pageItems.length, state.filtered.length);
  els.resultRange.textContent = state.filtered.length ? `${start + 1}-${end} of ${state.filtered.length}` : "0 results";
  els.pageLabel.textContent = `Page ${state.page} of ${Math.max(1, Math.ceil(state.filtered.length / PAGE_SIZE))}`;
  els.prevPage.disabled = state.page <= 1;
  els.nextPage.disabled = state.page >= Math.ceil(state.filtered.length / PAGE_SIZE);
}

function renderDetail() {
  const article = state.articles[state.selectedIndex];
  if (!article) {
    els.detailPanel.innerHTML = `
      <div class="empty-state">
        <h2>No article selected</h2>
        <p>Adjust the filters or load a dataset with matching articles.</p>
      </div>`;
    return;
  }

  els.detailPanel.innerHTML = `
    <h2 class="detail-title">${escapeHtml(article.title)}</h2>
    <p class="detail-meta">${escapeHtml(formatDate(article.timestamp))} | ${escapeHtml(
      article.authors.join(", ") || "Unknown author",
    )}</p>
    <div class="chip-list">
      ${article.tags.map((tag) => `<span class="chip">${escapeHtml(tag)}</span>`).join("")}
    </div>
    <div class="detail-actions">
      ${article.url ? `<a href="${escapeAttribute(article.url)}" target="_blank" rel="noreferrer">Open article</a>` : ""}
      <a class="secondary" href="data:text/plain;charset=utf-8,${encodeURIComponent(article.text)}" download="article.txt">Download text</a>
    </div>
    <div class="article-text">${escapeHtml(article.text)}</div>
  `;
}

function countBy(values) {
  return values.reduce((map, value) => {
    if (value) map.set(value, (map.get(value) || 0) + 1);
    return map;
  }, new Map());
}

function dateValue(article) {
  return article.timestamp?.getTime() ?? 0;
}

function getYear(article) {
  return article.timestamp?.getUTCFullYear();
}

function formatDate(date) {
  if (!date) return "Unknown date";
  return new Intl.DateTimeFormat("en", { year: "numeric", month: "short", day: "numeric" }).format(date);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return entities[char];
  });
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

els.datasetSelect.addEventListener("change", () => {
  loadDataset(els.datasetSelect.value).catch((error) => {
    els.datasetStatus.textContent = error.message;
  });
});

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const csv = await file.text();
  setArticles(parseCsv(csv), file.name);
});

[els.searchInput, els.tagFilter, els.yearFilter, els.sortSelect].forEach((input) => {
  input.addEventListener("input", () => {
    state.page = 1;
    applyFilters();
  });
});

els.prevPage.addEventListener("click", () => {
  state.page -= 1;
  renderArticles();
});

els.nextPage.addEventListener("click", () => {
  state.page += 1;
  renderArticles();
});

loadDataset(els.datasetSelect.value).catch((error) => {
  els.datasetStatus.textContent = error.message;
});
