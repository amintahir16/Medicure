/**
 * UI Rendering & Markdown Engine with Interactive Citation Badges
 */

export class MedicureUI {
  constructor() {
    this.speechSynth = window.speechSynthesis;
  }

  /**
   * Transforms raw markdown and inline citations into styled HTML
   */
  renderMarkdown(text, citations = []) {
    if (!text) return "";

    let html = text;

    // 0. Strip redundant duplicate text-based references sections dumped by LLMs
    html = html.replace(/\n*#{2,4}\s*(?:Verified\s+)?(?:Textbook\s+)?References[\s\S]*$/i, '');
    html = html.replace(/\n*\*{2}References:?\*{2}[\s\S]*$/i, '');

    // 1. Sanitize HTML tags except our custom spans
    html = html
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // 2. Headings (h4, h3, h2)
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

    // 3. Blockquotes / Clinical Pearls & Emergency Triage Alerts
    html = html.replace(/^&gt;\s*🚨\s*(.*?)$/gim, '<blockquote class="emergency-alert">🚨 $1</blockquote>');
    html = html.replace(/^&gt;\s*💡\s*(.*?)$/gim, '<blockquote>💡 $1</blockquote>');
    html = html.replace(/^&gt;\s*(.*?)$/gim, '<blockquote>$1</blockquote>');

    // 4. Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // 5. Tables (wrapped in responsive scrollable container)
    html = this._parseMarkdownTables(html);

    // 6. Ordered & Unordered Lists (with nested sub-bullet support)
    html = this._parseMarkdownLists(html);

    // 7. Paragraphs
    html = html
      .split(/\n\n+/)
      .map(para => {
        para = para.trim();
        if (para.startsWith('<h') || para.startsWith('<table') || 
            para.startsWith('<div class="table-responsive-wrapper"') ||
            para.startsWith('<ul') || para.startsWith('<ol') || para.startsWith('<blockquote')) {
          return para;
        }
        return `<p>${para.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    // 8. Transform Citations: [Ref X] into interactive badges
    html = this._replaceCitationBadges(html, citations);

    return html;
  }

  _parseMarkdownLists(text) {
    const lines = text.split('\n');
    let inUl = false;
    let inSubUl = false;
    let inOl = false;
    let result = [];

    const closeSubUl = () => {
      if (inSubUl) {
        result.push('</ul></li>');
        inSubUl = false;
      }
    };

    const closeAll = () => {
      closeSubUl();
      if (inUl) {
        result.push('</ul>');
        inUl = false;
      }
      if (inOl) {
        result.push('</ol>');
        inOl = false;
      }
    };

    for (let i = 0; i < lines.length; i++) {
      const rawLine = lines[i];
      const trimmed = rawLine.trim();

      // Check for indented sub-bullet: indented with 2+ spaces or tabs, starting with -, *, or •
      const subUlMatch = rawLine.match(/^(?:[ \t]{2,})[-*•]\s+(.*)$/);
      // Top level bullet
      const topUlMatch = trimmed.match(/^[-*•]\s+(.*)$/);
      // Numbered item
      const olMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);

      if (subUlMatch && (inUl || inOl)) {
        if (!inSubUl) {
          result.push('<ul class="sub-list">');
          inSubUl = true;
        }
        result.push(`<li>${subUlMatch[1]}</li>`);
      } else if (topUlMatch) {
        closeSubUl();
        if (inOl) {
          result.push('</ol>');
          inOl = false;
        }
        if (!inUl) {
          result.push('<ul>');
          inUl = true;
        }
        result.push(`<li>${topUlMatch[1]}</li>`);
      } else if (olMatch) {
        closeSubUl();
        if (inUl) {
          result.push('</ul>');
          inUl = false;
        }
        if (!inOl) {
          result.push('<ol>');
          inOl = true;
        }
        result.push(`<li>${olMatch[2]}</li>`);
      } else {
        closeAll();
        result.push(rawLine);
      }
    }
    closeAll();
    return result.join('\n');
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
          tableHtml = '<div class="table-responsive-wrapper"><table><thead><tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        }
      } else {
        if (inTable) {
          tableHtml += '</tbody></table></div>';
          result.push(tableHtml);
          inTable = false;
        }
        result.push(lines[i]);
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table></div>';
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

    // Pattern matching all bracket variants: [Ref 1], 【Ref 1】, 【Ref1】, Ref [1], ［Ref 1］, (Ref 1), etc.
    const pattern = /(?:[\[【［〔(]\s*Ref\.?\s*(\d+)(?::\s*([^\]】］〕)]+))?\s*[\]】］〕)]|Ref\.?\s*[\[【［〔(]\s*(\d+)\s*[\]】］〕)])/gi;

    return html.replace(pattern, (match, p1, p2, p3) => {
      const idx = parseInt(p1 || p3, 10);
      let cit = citMap[idx];

      // Smart fallback: if citMap[idx] is missing, use citations by index or first available citation
      if (!cit && citations && citations.length > 0) {
        if (citations[idx - 1]) {
          cit = citations[idx - 1];
        } else {
          cit = citations[0];
        }
      }

      let bookTitle = cit ? cit.book_title : "Textbook";
      let pageNum = cit ? cit.page_number : 1;
      let excerpt = cit ? (cit.excerpt || "") : "";

      if (!cit && p2) {
        const cleanDetails = p2.replace(/<[^>]+>/g, "").trim();
        const parts = cleanDetails.split("|").map(s => s.trim());
        bookTitle = parts[0] || "Textbook";
        const pageMatch = cleanDetails.match(/(?:p\.|page)\s*(\d+)/i);
        if (pageMatch) {
          pageNum = parseInt(pageMatch[1], 10);
        }
      }

      // Sleek, compact badge: Ref [1] • p. 6
      const badgeText = `Ref [${idx}] • p. ${pageNum}`;

      return `<button class="citation-pill" data-ref-idx="${idx}" data-book="${encodeURIComponent(bookTitle)}" data-page="${pageNum}" data-excerpt="${encodeURIComponent(excerpt)}" title="Inspect ${bookTitle}, Page ${pageNum}">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z"/></svg>
        ${badgeText}
      </button>`;
    });
  }

  /**
   * Renders the Verified Textbook References collapsible accordion
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
          <button class="ref-inspect-btn" data-book="${encodeURIComponent(c.book_title)}" data-page="${c.page_number}" data-excerpt="${encodeURIComponent(c.excerpt || '')}">
            📖 Inspect Page
          </button>
        </div>
      `;
    }).join("");

    const summarySources = citations.map(c => `${c.subject} p. ${c.page_number}`).join(", ");

    return `
      <div class="references-card">
        <div class="references-header toggle-refs-btn" title="Click to view verified curriculum page excerpts">
          <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <span>📚 <strong>${citations.length} Grounded MBBS ${citations.length === 1 ? 'Source' : 'Sources'}</strong></span>
            <span class="refs-summary-badge">${summarySources}</span>
          </div>
          <span class="refs-toggle-arrow">▾ View Sources</span>
        </div>
        <div class="references-list" style="display: none;">
          ${itemsHtml}
        </div>
      </div>
    `;
  }

  /**
   * Speaks the response aloud using Web Speech API
   */
  speakText(text, onEnd = null) {
    if (!this.speechSynth) return;
    if (this.speechSynth.speaking) {
      this.speechSynth.cancel();
      if (onEnd) onEnd();
      return;
    }
    // Clean markdown tags
    const plain = text.replace(/<[^>]*>/g, '').replace(/\[Ref.*?\]/g, '');
    const utterance = new SpeechSynthesisUtterance(plain);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    if (onEnd) {
      utterance.onend = onEnd;
      utterance.onerror = onEnd;
    }
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
