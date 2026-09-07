/**
 * UI Rendering & Markdown Engine with Interactive Citation Badges
 */

export class MedGeminiUI {
  constructor() {
    this.speechSynth = window.speechSynthesis;
  }

  /**
   * Transforms raw markdown and inline citations into styled HTML
   */
  renderMarkdown(text, citations = []) {
    if (!text) return "";

    let html = text;

    // 1. Sanitize HTML tags except our custom spans
    html = html
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // 2. Headings
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

    // 3. Blockquotes / Clinical Pearls
    html = html.replace(/^\&gt;\s*💡\s*(.*?)$/gim, '<blockquote>💡 $1</blockquote>');
    html = html.replace(/^\&gt;\s*(.*?)$/gim, '<blockquote>$1</blockquote>');

    // 4. Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // 5. Tables
    html = this._parseMarkdownTables(html);

    // 6. Bullet lists
    html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
    html = html.replace(/<\/ul>\s*<ul>/gim, ''); // Merge adjacent <ul>

    // 7. Paragraphs
    html = html
      .split(/\n\n+/)
      .map(para => {
        para = para.trim();
        if (para.startsWith('<h') || para.startsWith('<table') || 
            para.startsWith('<ul') || para.startsWith('<blockquote')) {
          return para;
        }
        return `<p>${para.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    // 8. Transform Citations: [Ref X: ... | Page Y] into interactive badges
    html = this._replaceCitationBadges(html, citations);

    return html;
  }

  _parseMarkdownTables(text) {
    const lines = text.split('\n');
    let inTable = false;
    let tableHtml = "";
    let result = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('|') && line.endsWith('|')) {
        // Skip markdown separator line |---|---|
        if (line.match(/^\|(?:\s*:?-+:?\s*\|)+$/)) {
          continue;
        }
        const cells = line.split('|').slice(1, -1).map(c => c.trim());
        if (!inTable) {
          inTable = true;
          tableHtml = '<table><thead><tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        }
      } else {
        if (inTable) {
          tableHtml += '</tbody></table>';
          result.push(tableHtml);
          inTable = false;
        }
        result.push(lines[i]);
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table>';
      result.push(tableHtml);
    }
    return result.join('\n');
  }

  _replaceCitationBadges(html, citations) {
    // Lookup dictionary by ref_index
    const citMap = {};
    (citations || []).forEach(c => {
      citMap[c.ref_index] = c;
    });

    // Pattern 1: [Ref X: Book Title | Chapter | Topic | Page Y]
    html = html.replace(/\[Ref\s*(\d+):?\s*([^\]]+)?\]/g, (match, idxStr, details) => {
      const idx = parseInt(idxStr, 10);
      const cit = citMap[idx];
      let bookTitle = cit ? cit.book_title : "MBBS Textbook";
      let pageNum = cit ? cit.page_number : 1;
      let subject = cit ? cit.subject : "";

      if (!cit && details) {
        // Strip HTML tags and isolate book name
        const cleanDetails = details.replace(/<[^>]+>/g, "").trim();
        const parts = cleanDetails.split("|").map(s => s.trim());
        bookTitle = parts[0] || "MBBS Textbook";
        
        // Try to extract page number from details (e.g. "p. 345", "Page 2")
        const pageMatch = cleanDetails.match(/(?:p\.|page)\s*(\d+)/i);
        if (pageMatch) {
          pageNum = parseInt(pageMatch[1], 10);
        }
      }

      const badgeText = subject ? `Ref [${idx}] • ${subject} p. ${pageNum}` : `Ref [${idx}] • p. ${pageNum}`;

      return `<button class="citation-pill" data-ref-idx="${idx}" data-book="${encodeURIComponent(bookTitle)}" data-page="${pageNum}" title="Click to inspect page ${pageNum} in ${bookTitle}">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z"/></svg>
        ${badgeText}
      </button>`;
    });

    return html;
  }

  /**
   * Renders the Verified Textbook References accordion at bottom of message
   */
  renderReferencesCard(citations) {
    if (!citations || citations.length === 0) return "";

    const itemsHtml = citations.map(c => {
      return `
        <div class="reference-item">
          <div class="ref-info">
            <div class="ref-book-line">
              <span class="ref-badge-index">REF ${c.ref_index}</span>
              <span>${c.book_title}</span>
            </div>
            <div class="ref-meta-line">
              <strong>${c.subject} (${c.mbbs_year})</strong> • ${c.chapter} • <strong>Page ${c.page_number}</strong> of ${c.total_pages}
            </div>
            <div class="ref-quote-box">"${c.excerpt || ''}"</div>
          </div>
          <button class="ref-inspect-btn" data-book="${encodeURIComponent(c.book_title)}" data-page="${c.page_number}">
            📖 Inspect Page
          </button>
        </div>
      `;
    }).join("");

    return `
      <div class="references-card">
        <div class="references-header">
          <span>📚 Verified Textbook Evidence Sources (${citations.length} Books Grounded)</span>
          <span style="font-size: 0.72rem; color: var(--med-blue);">Click any source to view full page</span>
        </div>
        <div class="references-list">
          ${itemsHtml}
        </div>
      </div>
    `;
  }

  /**
   * Speaks the response aloud using Web Speech API
   */
  speakText(text) {
    if (!this.speechSynth) return;
    if (this.speechSynth.speaking) {
      this.speechSynth.cancel();
      return;
    }
    // Clean markdown tags
    const plain = text.replace(/<[^>]*>/g, '').replace(/\[Ref.*?\]/g, '');
    const utterance = new SpeechSynthesisUtterance(plain);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    this.speechSynth.speak(utterance);
  }

  /**
   * Copies text to clipboard with tooltip confirmation
   */
  async copyToClipboard(text, buttonElement) {
    try {
      await navigator.clipboard.writeText(text);
      const originalHtml = buttonElement.innerHTML;
      buttonElement.innerHTML = "✓ Copied";
      setTimeout(() => {
        buttonElement.innerHTML = originalHtml;
      }, 1800);
    } catch (e) {
      console.error("Clipboard copy failed", e);
    }
  }
}
