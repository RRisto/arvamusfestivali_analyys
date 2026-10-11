(function () {
  "use strict";

  const page = document.body.dataset.page;
  const activeLink = document.querySelector(`[data-nav="${page}"]`);
  if (activeLink) activeLink.setAttribute("aria-current", "page");

  window.siteCommon = {
    async fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Andmete laadimine ebaõnnestus (${response.status}).`);
      return response.json();
    },
    showError(error) {
      const target = document.getElementById("page-error");
      if (!target) return;
      target.textContent = error instanceof Error ? error.message : String(error);
      target.hidden = false;
    },
    formatNumber(value, digits = 0) {
      return new Intl.NumberFormat("et-EE", { maximumFractionDigits: digits }).format(value);
    },
    plotConfig: { responsive: true, displaylogo: false, scrollZoom: true },
    chartLayout: {
      font: { family: "Inter, system-ui, sans-serif", color: "#17322b", size: 13 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      margin: { l: 42, r: 24, t: 55, b: 52 },
    },
    observeOnce(element, callback) {
      element.classList.add("chart-loading");
      element.textContent = "Joonise laadimine…";
      if (!("IntersectionObserver" in window)) {
        callback();
        return;
      }
      const observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        callback();
      }, { rootMargin: "240px" });
      observer.observe(element);
    },
  };
})();
