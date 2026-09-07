/**
 * Reference & PDF Page Inspector Drawer
 * Allows medical students to inspect the exact referenced textbook page,
 * view highlighted clinical quotes, and navigate adjacent pages.
 */

export class MedGeminiBookViewer {
  constructor(api, drawerElement) {
    this.api = api;
    this.drawer = drawerElement;
    this.currentBook = null;
    this.currentPage = 1;
    this.totalPages = 1;
    this.currentExcerpt = "";

    this.initElements();
  }

  initElements() {
    this.titleEl = this.drawer.querySelector(".inspector-title");
    this.closeBtn = this.drawer.querySelector(".close-inspector-btn");
    this.metaEl = this.drawer.querySelector(".inspector-book-meta");
    this.viewerEl = this.drawer.querySelector(".page-content-viewer");
    this.pageIndicator = this.drawer.querySelector(".page-indicator");
    this.prevBtn = this.drawer.querySelector(".prev-page-btn");
    this.nextBtn = this.drawer.querySelector(".next-page-btn");

    if (this.closeBtn) {
      this.closeBtn.addEventListener("click", () => this.close());
    }

    if (this.prevBtn) {
      this.prevBtn.addEventListener("click", () => {
        if (this.currentPage > 1) {
          this.loadPage(this.currentBook, this.currentPage - 1, "");
        }
      });
    }

    if (this.nextBtn) {
      this.nextBtn.addEventListener("click", () => {
        if (this.currentPage < this.totalPages) {
          this.loadPage(this.currentBook, this.currentPage + 1, "");
        }
      });
    }

    // Close on Escape key
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isOpen()) {
        this.close();
      }
    });
  }

  isOpen() {
    return this.drawer.classList.contains("open");
  }

  open() {
    this.drawer.classList.add("open");
  }

  close() {
    this.drawer.classList.remove("open");
  }

  async loadPage(bookTitle, pageNumber, highlightExcerpt = "") {
    this.currentBook = bookTitle;
    this.currentPage = parseInt(pageNumber, 10);
    this.currentExcerpt = highlightExcerpt;

    this.open();
    this.viewerEl.innerHTML = `<div style="text-align:center; padding: 40px; color: var(--text-muted);">
      <div class="typing-indicator" style="justify-content:center; margin-bottom: 12px;">
        <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
      </div>
      Loading Page ${pageNumber} of ${bookTitle}...
    </div>`;

    try {
      const data = await this.api.getPageContent(bookTitle, this.currentPage);
      this.renderPageData(data);
    } catch (e) {
      this.viewerEl.innerHTML = `
        <div style="padding: 24px; color: var(--med-rose); text-align: center;">
          <p>⚠️ Unable to load textbook page.</p>
          <p style="font-size: 0.8rem; margin-top: 6px; color: var(--text-muted);">${e.message}</p>
        </div>
      `;
    }
  }

  renderPageData(data) {
    this.totalPages = data.total_pages || 1;
    this.titleEl.textContent = `Textbook Reference Viewer`;
    
    this.metaEl.innerHTML = `
      <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">
        ${data.book_title}
      </div>
      <div style="font-size: 0.78rem; color: var(--med-blue); margin-bottom: 4px;">
        <strong>${data.subject}</strong> • ${data.mbbs_year}
      </div>
      <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 2px;">
        <strong>Chapter:</strong> ${data.chapter}
      </div>
      <div style="font-size: 0.8rem; color: var(--text-secondary);">
        <strong>Topic:</strong> ${data.topic}
      </div>
    `;

    this.pageIndicator.textContent = `Page ${data.page_number} of ${this.totalPages}`;
    this.prevBtn.disabled = this.currentPage <= 1;
    this.nextBtn.disabled = this.currentPage >= this.totalPages;

    let content = data.content || "";
    
    // Highlight matching excerpt if available
    if (this.currentExcerpt && this.currentExcerpt.length > 20) {
      const cleanQuote = this.currentExcerpt.slice(0, 80).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      try {
        const regex = new RegExp(`(${cleanQuote})`, 'i');
        content = content.replace(regex, '<mark>$1</mark>');
      } catch (e) {
        // Fallback without highlighting if regex fails
      }
    }

    this.viewerEl.innerHTML = content;
  }
}
