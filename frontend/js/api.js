/**
 * API Client for Medicure Backend
 */

export class MedicureAPI {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
  }

  async checkHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error("Health check failed");
    return res.json();
  }

  async sendChatMessage(payload) {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Chat request failed" }));
      throw new Error(err.detail || "Error connecting to Medicure");
    }
    return res.json();
  }

  async getBooks() {
    const res = await fetch(`${this.baseUrl}/api/books`);
    if (!res.ok) throw new Error("Failed to fetch books");
    return res.json();
  }

  async getPageContent(bookTitle, pageNumber) {
    const params = new URLSearchParams({
      book_title: bookTitle,
      page_number: pageNumber.toString(),
    });
    const res = await fetch(`${this.baseUrl}/api/books/page?${params.toString()}`);
    if (!res.ok) throw new Error("Page not found");
    return res.json();
  }

  async uploadBook(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${this.baseUrl}/api/books/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || "Failed to upload PDF");
    }
    return res.json();
  }

  async getSamplePrompts() {
    const res = await fetch(`${this.baseUrl}/api/sample-prompts`);
    if (!res.ok) throw new Error("Failed to fetch prompts");
    return res.json();
  }

  async getSessions() {
    const res = await fetch(`${this.baseUrl}/api/sessions`);
    if (!res.ok) throw new Error("Failed to fetch sessions");
    return res.json();
  }

  async getSessionMessages(sessionId) {
    const res = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`);
    if (!res.ok) throw new Error("Failed to fetch session messages");
    return res.json();
  }

  async createSession(title, mbbsYear, subject) {
    const res = await fetch(`${this.baseUrl}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, mbbs_year: mbbsYear, subject }),
    });
    if (!res.ok) throw new Error("Failed to create session");
    return res.json();
  }

  async deleteSession(sessionId) {
    const res = await fetch(`${this.baseUrl}/api/sessions/${sessionId}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete session");
    return res.json();
  }
}
