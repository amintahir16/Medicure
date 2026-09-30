/**
 * Medicure Application Main Controller
 */

import { MedicureAPI } from "./api.js";
import { MedicureUI } from "./ui.js";
import { MedicureBookViewer } from "./book_viewer.js";

// Storage helper for seamless continuity between Medicure and legacy keys
// Storage helper for Medicure with legacy migration support
const getStorage = (key) => {
  const current = localStorage.getItem(`medicure_${key}`);
  if (current !== null) return current;
  const legacy = localStorage.getItem(`medgemini_${key}`);
  if (legacy !== null) {
    localStorage.setItem(`medicure_${key}`, legacy);
    localStorage.removeItem(`medgemini_${key}`);
    return legacy;
  }
  return null;
};
const setStorage = (key, val) => {
  localStorage.setItem(`medicure_${key}`, val);
};
const removeStorage = (key) => {
  localStorage.removeItem(`medicure_${key}`);
  localStorage.removeItem(`medgemini_${key}`);
};

class MedicureApp {
  constructor() {
    this.api = new MedicureAPI();
    this.ui = new MedicureUI();
    this.sessionId = getStorage("session_id") || null;
    this.currentYearFilter = "all";
    this.currentSubjectFilter = "all";
    this.currentStudyMode = "standard";

    // User settings
    this.settings = {
      provider: getStorage("provider") || "offline",
      geminiKey: getStorage("gemini_key") || "",
      groqKey: getStorage("groq_key") || "",
      theme: getStorage("theme") || "dark"
    };

    this.init();
  }

  async init() {
    this.initDOMElements();
    this.applyTheme(this.settings.theme);
    this.updateModelStatusBadge();
    this.initBookViewer();
    this.attachEventListeners();
    await this.loadInitialData();
  }

  initDOMElements() {
    // Layout elements
    this.sidebar = document.getElementById("sidebar");
    this.toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
    this.newChatBtn = document.getElementById("new-chat-btn");
    this.historyList = document.getElementById("history-list");
    this.chatScrollArea = document.getElementById("chat-scroll-area");
    this.welcomeHero = document.getElementById("welcome-hero");
    this.promptTextarea = document.getElementById("prompt-textarea");
    this.sendBtn = document.getElementById("send-btn");
    this.themeToggleBtn = document.getElementById("theme-toggle-btn");
    this.modelPillBtn = document.getElementById("model-pill-btn");
    this.modelPillText = document.getElementById("model-pill-text");
    this.modelStatusDot = document.getElementById("model-status-dot");
    this.headerLibraryBtn = document.getElementById("header-library-btn");
    this.headerSettingsBtn = document.getElementById("header-settings-btn");
    this.headerBookCount = document.getElementById("header-book-count");
    this.toastContainer = document.getElementById("toast-container");

    // Filters & Selectors
    this.yearFilterSelect = document.getElementById("year-filter-select");
    this.studyModeSelect = document.getElementById("study-mode-select");
    this.subjectPills = document.querySelectorAll(".subject-pill");

    // Modals
    this.settingsModal = document.getElementById("settings-modal");
    this.settingsBtn = document.getElementById("settings-btn");
    this.closeSettingsBtn = document.getElementById("close-settings-btn");
    this.saveSettingsBtn = document.getElementById("save-settings-btn");
    this.providerRadios = document.querySelectorAll("input[name='llm-provider']");
    this.geminiKeyInput = document.getElementById("gemini-key-input");
    this.groqKeyInput = document.getElementById("groq-key-input");

    this.libraryModal = document.getElementById("library-modal");
    this.libraryBtn = document.getElementById("library-btn");
    this.closeLibraryBtn = document.getElementById("close-library-btn");
    this.booksListContainer = document.getElementById("books-list-container");
    this.dropzone = document.getElementById("dropzone");
    this.fileInput = document.getElementById("pdf-file-input");
  }

  initBookViewer() {
    const drawerEl = document.getElementById("inspector-drawer");
    this.bookViewer = new MedicureBookViewer(this.api, drawerEl);
  }

  attachEventListeners() {
    // Sidebar toggle
    this.toggleSidebarBtn.addEventListener("click", () => {
      this.sidebar.classList.toggle("collapsed");
    });

    // New Consultation
    this.newChatBtn.addEventListener("click", () => this.startNewChat());

    // Prompt Textarea Auto-grow & Keydown / Enter handling
    const updateSendState = () => {
      this.promptTextarea.style.height = "auto";
      this.promptTextarea.style.height = Math.min(this.promptTextarea.scrollHeight, 160) + "px";
      if (this.sendBtn) {
        this.sendBtn.disabled = !this.promptTextarea.value.trim();
      }
    };

    this.promptTextarea.addEventListener("input", updateSendState);
    this.promptTextarea.addEventListener("keyup", updateSendState);
    this.promptTextarea.addEventListener("change", updateSendState);

    this.promptTextarea.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.keyCode === 13) && !e.shiftKey) {
        e.preventDefault();
        const text = this.promptTextarea.value.trim();
        if (text) {
          this.handleSendMessage();
        }
      }
    });

    if (this.sendBtn) {
      this.sendBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const text = this.promptTextarea.value.trim();
        if (text) {
          this.handleSendMessage();
        }
      });
    }

    // Year & Study Mode Selectors
    this.yearFilterSelect.addEventListener("change", (e) => {
      this.currentYearFilter = e.target.value;
    });

    this.studyModeSelect.addEventListener("change", (e) => {
      this.currentStudyMode = e.target.value;
    });

    // Subject Pills with smooth scroll to center
    this.subjectPills.forEach(pill => {
      pill.addEventListener("click", () => {
        this.subjectPills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        this.currentSubjectFilter = pill.dataset.subject;
        pill.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      });
    });

    // Theme Toggle
    this.themeToggleBtn.addEventListener("click", () => {
      const nextTheme = this.settings.theme === "dark" ? "light" : "dark";
      this.applyTheme(nextTheme);
    });

    // Model Status Pill Click -> Opens Settings Modal
    if (this.modelPillBtn) {
      this.modelPillBtn.addEventListener("click", () => this.openSettingsModal());
      this.modelPillBtn.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.openSettingsModal();
        }
      });
    }

    // Top Header Buttons
    if (this.headerSettingsBtn) {
      this.headerSettingsBtn.addEventListener("click", () => this.openSettingsModal());
    }
    if (this.headerLibraryBtn) {
      this.headerLibraryBtn.addEventListener("click", () => this.openLibraryModal());
    }

    // Settings Modal
    this.settingsBtn.addEventListener("click", () => this.openSettingsModal());
    this.closeSettingsBtn.addEventListener("click", () => this.settingsModal.classList.remove("active"));
    this.saveSettingsBtn.addEventListener("click", () => this.saveSettings());

    // Library Modal
    this.libraryBtn.addEventListener("click", () => this.openLibraryModal());
    this.closeLibraryBtn.addEventListener("click", () => this.libraryModal.classList.remove("active"));

    // Drag-and-drop file upload
    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.uploadPDF(e.target.files[0]);
      }
    });

    this.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropzone.classList.add("dragover");
    });

    this.dropzone.addEventListener("dragleave", () => {
      this.dropzone.classList.remove("dragover");
    });

    this.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        this.uploadPDF(e.dataTransfer.files[0]);
      }
    });

    // Delegated click for Citation Pills & Reference Inspect buttons
    document.addEventListener("click", (e) => {
      const citPill = e.target.closest(".citation-pill");
      if (citPill) {
        const bookTitle = decodeURIComponent(citPill.dataset.book);
        const pageNum = citPill.dataset.page;
        const excerpt = decodeURIComponent(citPill.dataset.excerpt || "");
        this.bookViewer.loadPage(bookTitle, pageNum, excerpt);
        return;
      }

      const inspectBtn = e.target.closest(".ref-inspect-btn");
      if (inspectBtn) {
        const bookTitle = decodeURIComponent(inspectBtn.dataset.book);
        const pageNum = inspectBtn.dataset.page;
        const excerpt = decodeURIComponent(inspectBtn.dataset.excerpt || "");
        this.bookViewer.loadPage(bookTitle, pageNum, excerpt);
        return;
      }

      const refsToggle = e.target.closest(".toggle-refs-btn");
      if (refsToggle) {
        const card = refsToggle.closest(".references-card");
        const list = card ? card.querySelector(".references-list") : null;
        const arrow = refsToggle.querySelector(".refs-toggle-arrow");
        if (list) {
          const isHidden = list.style.display === "none";
          list.style.display = isHidden ? "flex" : "none";
          if (arrow) {
            arrow.innerHTML = isHidden ? "▴ Hide Sources" : "▾ View Sources";
          }
        }
        return;
      }

      const copyBtn = e.target.closest(".copy-msg-btn");
      if (copyBtn) {
        const textToCopy = decodeURIComponent(copyBtn.dataset.rawText || "");
        this.ui.copyToClipboard(textToCopy, copyBtn);
        this.showToast("📋 Copied clinical answer to notes", "info");
        return;
      }

      const speakBtn = e.target.closest(".speak-msg-btn");
      if (speakBtn) {
        const textToSpeak = decodeURIComponent(speakBtn.dataset.rawText || "");
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
          window.speechSynthesis.cancel();
          document.querySelectorAll(".speak-msg-btn").forEach(b => {
            b.classList.remove("active-speaking");
            b.innerHTML = "🔊 Read Aloud";
          });
        } else {
          document.querySelectorAll(".speak-msg-btn").forEach(b => {
            b.classList.remove("active-speaking");
            b.innerHTML = "🔊 Read Aloud";
          });
          speakBtn.classList.add("active-speaking");
          speakBtn.innerHTML = "⏹️ Stop";
          this.ui.speakText(textToSpeak, () => {
            speakBtn.classList.remove("active-speaking");
            speakBtn.innerHTML = "🔊 Read Aloud";
          });
        }
        return;
      }
    });
  }

  applyTheme(theme) {
    this.settings.theme = theme;
    setStorage("theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
    if (this.themeToggleBtn) {
      this.themeToggleBtn.innerHTML = theme === "dark" ? "☀️ Light" : "🌙 Dark";
    }
  }

  async loadInitialData() {
    try {
      const health = await this.api.checkHealth();
      this.updateModelStatusBadge(health);
      await this.loadSamplePrompts();
      await this.loadHistorySessions();

      // If user had a previously opened consultation, restore it
      if (this.sessionId) {
        await this.loadSession(this.sessionId);
      }
    } catch (e) {
      console.error("Initialization error:", e);
    }
  }

  updateModelStatusBadge(health = null) {
    if (!this.modelPillText || !this.modelStatusDot) return;
    const bookCount = (health && health.indexed_books_count !== undefined) ? health.indexed_books_count : 1;
    const bookLabel = bookCount === 1 ? "1 Book" : `${bookCount} Books`;

    if (this.headerBookCount) {
      this.headerBookCount.textContent = bookCount;
    }
    const sidebarCountEl = document.getElementById("sidebar-book-count");
    if (sidebarCountEl) {
      sidebarCountEl.textContent = bookLabel;
    }

    if (this.settings.provider === "gemini") {
      const hasKey = !!this.settings.geminiKey;
      this.modelPillText.textContent = hasKey ? "Google Gemini 2.0 Flash" : "Google Gemini (Key Required)";
      this.modelStatusDot.style.backgroundColor = "#3b82f6";
    } else if (this.settings.provider === "groq") {
      const hasKey = !!this.settings.groqKey;
      this.modelPillText.textContent = hasKey ? "Groq Cloud AI (OpenAI GPT-OSS 120B)" : "Groq Cloud AI (Key Required)";
      this.modelStatusDot.style.backgroundColor = "#8b5cf6";
    } else {
      this.modelPillText.textContent = `Medicure Grounded Engine (${bookLabel})`;
      this.modelStatusDot.style.backgroundColor = "#10b981";
    }
  }

  async loadSamplePrompts() {
    try {
      const data = await this.api.getSamplePrompts();
      const grid = document.getElementById("prompt-grid");
      if (!grid) return;

      grid.innerHTML = data.prompts.map(p => `
        <div class="prompt-card" data-query="${encodeURIComponent(p.query)}">
          <div class="card-header-row">
            <span class="card-category">${p.category}</span>
            <span class="card-year">${p.year}</span>
          </div>
          <div class="card-title">${p.title}</div>
          <div class="card-text">${p.query}</div>
        </div>
      `).join("");

      grid.querySelectorAll(".prompt-card").forEach(card => {
        card.addEventListener("click", () => {
          const query = decodeURIComponent(card.dataset.query);
          this.promptTextarea.value = query;
          this.promptTextarea.style.height = "auto";
          this.sendBtn.disabled = false;
          this.handleSendMessage();
        });
      });
    } catch (e) {
      console.error("Failed to load prompts", e);
    }
  }

  async loadHistorySessions() {
    try {
      const sessions = await this.api.getSessions();
      if (!this.historyList) return;

      if (!sessions || sessions.length === 0) {
        this.historyList.innerHTML = `<div style="font-size:0.75rem; color:var(--text-muted); padding:8px 12px;">No prior consultations.</div>`;
        return;
      }

      this.historyList.innerHTML = sessions.map(s => `
        <div class="history-item ${s.session_id === this.sessionId ? 'active' : ''}" data-session-id="${s.session_id}">
          <svg class="history-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>
          <span class="history-title" title="${this.escapeHtml(s.title)}">${this.escapeHtml(s.title)}</span>
          <button class="delete-session-btn" data-session-id="${s.session_id}" title="Delete consultation">✕</button>
        </div>
      `).join("");

      this.historyList.querySelectorAll(".history-item").forEach(item => {
        item.addEventListener("click", (e) => {
          if (e.target.closest(".delete-session-btn")) return;
          this.loadSession(item.dataset.sessionId);
        });
      });

      this.historyList.querySelectorAll(".delete-session-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const sid = btn.dataset.sessionId;
          try {
            await this.api.deleteSession(sid);
            if (this.sessionId === sid) {
              this.startNewChat();
            }
            await this.loadHistorySessions();
            this.showToast("Consultation removed", "info");
          } catch (err) {
            console.error("Failed to delete session", err);
          }
        });
      });
    } catch (e) {
      console.error("Failed to load history sessions", e);
    }
  }

  async loadSession(sessionId) {
    this.sessionId = sessionId;
    setStorage("session_id", sessionId);

    this.historyList?.querySelectorAll(".history-item").forEach(item => {
      item.classList.toggle("active", item.dataset.sessionId === sessionId);
    });

    try {
      const data = await this.api.getSessionMessages(sessionId);
      
      // Clear current messages
      const existingRows = this.chatScrollArea.querySelectorAll(".message-row");
      existingRows.forEach(r => r.remove());

      if (!data.messages || data.messages.length === 0) {
        this.welcomeHero.style.display = "flex";
        return;
      }

      this.welcomeHero.style.display = "none";

      data.messages.forEach(msg => {
        if (msg.role === "user") {
          this.appendUserMessage(msg.content);
        } else {
          this.appendAIMessage(msg.content, msg.citations, msg.provider_used);
        }
      });

      this.scrollToBottom();
    } catch (e) {
      console.error("Error loading session messages:", e);
      this.startNewChat();
    }
  }

  startNewChat() {
    this.sessionId = null;
    removeStorage("session_id");
    const existingRows = this.chatScrollArea.querySelectorAll(".message-row");
    existingRows.forEach(r => r.remove());
    this.welcomeHero.style.display = "flex";
    this.historyList.querySelectorAll(".history-item").forEach(item => item.classList.remove("active"));
  }

  async handleSendMessage() {
    const query = this.promptTextarea.value.trim();
    if (!query) return;

    // Reset input
    this.promptTextarea.value = "";
    this.promptTextarea.style.height = "auto";
    this.sendBtn.disabled = true;

    // Hide hero if visible
    if (this.welcomeHero.style.display !== "none") {
      this.welcomeHero.style.display = "none";
    }

    // Append User Message
    this.appendUserMessage(query);
    this.scrollToBottom();

    // Show AI typing indicator
    const loadingRow = this.appendLoadingMessage();
    this.scrollToBottom();

    try {
      const payload = {
        query: query,
        session_id: this.sessionId,
        provider: this.settings.provider,
        api_key: this.settings.provider === "gemini" ? this.settings.geminiKey : this.settings.groqKey,
        year_filter: this.currentYearFilter,
        subject_filter: this.currentSubjectFilter,
        study_mode: this.currentStudyMode
      };

      const resp = await this.api.sendChatMessage(payload);

      // Always update active session id & refresh recent consultations list
      this.sessionId = resp.session_id;
      setStorage("session_id", this.sessionId);
      await this.loadHistorySessions();

      // Remove loading indicator & display AI message
      loadingRow.remove();
      this.appendAIMessage(resp.answer, resp.citations, resp.provider_used);
      this.scrollToBottom();

    } catch (e) {
      loadingRow.remove();
      this.appendAIMessage(
        `⚠️ **Error generating medical response**: ${e.message}. Please verify your API key or model settings.`,
        [],
        "Error Handler"
      );
      this.scrollToBottom();
    }
  }

  escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="message-content-wrapper user-wrapper">
        <div class="user-message-bubble">${this.escapeHtml(text)}</div>
      </div>
    `;
    this.chatScrollArea.appendChild(row);
    return row;
  }

  appendAIMessage(rawAnswer, citations = [], providerUsed = "offline") {
    const row = document.createElement("div");
    row.className = "message-row ai-row";

    const formattedHtml = this.ui.renderMarkdown(rawAnswer, citations);
    const referencesHtml = this.ui.renderReferencesCard(citations);

    let providerLabel = "📚 Grounded Engine";
    if (providerUsed === "groq" || providerUsed.toLowerCase().includes("groq")) {
      providerLabel = "⚡ Groq (GPT-OSS 120B)";
    } else if (providerUsed === "gemini" || providerUsed.toLowerCase().includes("gemini")) {
      providerLabel = "✨ Gemini 2.0";
    }

    row.innerHTML = `
      <div class="message-avatar avatar-ai">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25L19 9zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5z"/></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="ai-message-body">
          ${formattedHtml}
        </div>
        ${referencesHtml}
        <div class="message-actions">
          <button class="msg-action-btn copy-msg-btn" data-raw-text="${encodeURIComponent(rawAnswer)}" title="Copy response to notes">
            📋 Copy
          </button>
          <button class="msg-action-btn speak-msg-btn" data-raw-text="${encodeURIComponent(rawAnswer)}" title="Listen to response">
            🔊 Read Aloud
          </button>
          <span class="model-provider-badge">
            ${providerLabel}
          </span>
        </div>
      </div>
    `;
    this.chatScrollArea.appendChild(row);
    return row;
  }

  appendLoadingMessage() {
    const row = document.createElement("div");
    row.className = "message-row";
    row.innerHTML = `
      <div class="message-avatar avatar-ai">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25L19 9zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5z"/></svg>
      </div>
      <div class="message-content-wrapper">
        <div class="ai-message-body" style="color: var(--text-muted); font-size: 0.88rem; display: flex; align-items: center; gap: 8px;">
          <div class="typing-indicator">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
          </div>
          <span>Cross-referencing 5 years of MBBS textbooks...</span>
        </div>
      </div>
    `;
    this.chatScrollArea.appendChild(row);
    return row;
  }

  scrollToBottom() {
    this.chatScrollArea.scrollTop = this.chatScrollArea.scrollHeight;
  }

  openSettingsModal() {
    this.providerRadios.forEach(r => {
      r.checked = r.value === this.settings.provider;
    });
    this.geminiKeyInput.value = this.settings.geminiKey;
    this.groqKeyInput.value = this.settings.groqKey;
    this.settingsModal.classList.add("active");
  }

  saveSettings() {
    const selectedProvider = Array.from(this.providerRadios).find(r => r.checked)?.value || "offline";
    this.settings.provider = selectedProvider;
    this.settings.geminiKey = this.geminiKeyInput.value.trim();
    this.settings.groqKey = this.groqKeyInput.value.trim();

    setStorage("provider", this.settings.provider);
    setStorage("gemini_key", this.settings.geminiKey);
    setStorage("groq_key", this.settings.groqKey);

    this.settingsModal.classList.remove("active");
    this.updateModelStatusBadge();
    this.showToast("Settings & model engine updated!", "success");
  }

  showToast(message, type = "info") {
    if (!this.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    this.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      toast.style.transition = "all 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, 2800);
  }


  async openLibraryModal() {
    this.libraryModal.classList.add("active");
    this.booksListContainer.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text-muted);">Loading library...</div>`;

    try {
      const data = await this.api.getBooks();
      this.booksListContainer.innerHTML = data.books.map(b => `
        <div style="background-color:var(--bg-tertiary); border:1px solid var(--border-subtle); border-radius:10px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-weight:600; font-size:0.88rem; color:var(--text-primary);">${b.book_title}</div>
            <div style="font-size:0.75rem; color:var(--med-blue); margin-top:2px;">
              ${b.subject} • ${b.mbbs_year}
            </div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:2px;">
              ${b.total_pages} Pages • ${b.indexed_chunks} Indexed Sections
            </div>
          </div>
          <button class="ref-inspect-btn" data-book="${encodeURIComponent(b.book_title)}" data-page="1">
            Read Book
          </button>
        </div>
      `).join("");
    } catch (e) {
      this.booksListContainer.innerHTML = `<div style="color:var(--med-rose); padding:10px;">Failed to load books: ${e.message}</div>`;
    }
  }

  async uploadPDF(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please select a valid PDF file.");
      return;
    }

    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const dropzoneText = this.dropzone.querySelector("div");
    const originalHtml = dropzoneText.innerHTML;
    dropzoneText.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
        <div class="typing-indicator" style="justify-content:center;">
          <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>
        <strong>Uploading & indexing ${file.name} (${fileSizeMB} MB)...</strong>
        <span style="font-size:0.75rem; color:var(--text-muted);">Extracting chapters, topics, and physical pages for curriculum grounding...</span>
      </div>
    `;

    try {
      const res = await this.api.uploadBook(file);
      alert(`Success! Indexed '${file.name}' with ${res.indexed_chunks} pages/sections.`);
      await this.openLibraryModal();
      await this.loadInitialData();
    } catch (e) {
      alert(`Upload failed: ${e.message}`);
    } finally {
      dropzoneText.innerHTML = originalHtml;
    }
  }
}

// Instantiate on DOM load
document.addEventListener("DOMContentLoaded", () => {
  window.medicure = new MedicureApp();
  window.medGemini = window.medicure;
});
