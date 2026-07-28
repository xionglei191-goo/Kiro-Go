package proxy

import (
	"encoding/json"
	"kiro-go/config"
	"net/http"
	"strconv"
	"time"
)

// Customer-facing "/api/*" endpoints. These are authenticated with the caller's
// own API key (Authorization: Bearer <key> or X-Api-Key header) and only ever
// expose data scoped to that key:
//
//	GET /api/stats — per-key usage counters (like /v1/stats but key-scoped)
//	GET /api/me    — quota/limit view for the key
//	GET /api/logs  — recent request logs attributed to the key
//
// Unlike inference endpoints, these read-only endpoints intentionally accept
// disabled or over-quota keys: a customer whose key was auto-deactivated after
// exhausting its credits must still be able to check the remaining quota (0)
// and see their history. The key just has to EXIST in the configured entries.

// customerKeyStatus derives a display status for an API key entry:
// "exhausted" when a limit is hit, "disabled" when manually turned off,
// otherwise "active". Exhausted wins over disabled because auto-deactivation
// sets Enabled=false when quota runs out and "exhausted" is the more useful signal.
func customerKeyStatus(e config.ApiKeyEntry) string {
	if overToken, overCredit := config.ApiKeyOverLimit(e); overToken || overCredit {
		return "exhausted"
	}
	if !e.Enabled {
		return "disabled"
	}
	return "active"
}

// creditsRemaining computes the remaining credit allowance for an entry.
// A zero CreditLimit means unlimited, reported as -1 so clients can distinguish
// "no limit" from "nothing left".
func creditsRemaining(e config.ApiKeyEntry) float64 {
	if e.CreditLimit <= 0 {
		return -1
	}
	rem := e.CreditLimit - e.CreditsUsed
	if rem < 0 {
		rem = 0
	}
	return rem
}

// tokensRemaining is the token-limit analogue of creditsRemaining (-1 = unlimited).
func tokensRemaining(e config.ApiKeyEntry) int64 {
	if e.TokenLimit <= 0 {
		return -1
	}
	rem := e.TokenLimit - e.TokensUsed
	if rem < 0 {
		rem = 0
	}
	return rem
}

// authenticateCustomerKey resolves the caller's API key to a configured entry.
// Writes a JSON 401 and returns nil when the key is missing or unknown.
// It deliberately does NOT check Enabled/limits — see the package comment above.
func (h *Handler) authenticateCustomerKey(w http.ResponseWriter, r *http.Request) *config.ApiKeyEntry {
	provided := extractProvidedKey(r)
	if provided == "" {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{"error": "Missing API key"})
		return nil
	}
	entry := config.FindApiKeyByValue(provided)
	if entry == nil {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid API key"})
		return nil
	}
	return entry
}

// handleCustomerStats GET /api/stats — usage counters scoped to the caller's key.
// Mirrors the shape of /v1/stats but every number is per-key, not global.
func (h *Handler) handleCustomerStats(w http.ResponseWriter, r *http.Request) {
	entry := h.authenticateCustomerKey(w, r)
	if entry == nil {
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		// "status" is the envelope health field (mirrors /v1/stats); the key's
		// own lifecycle state therefore lives under "keyStatus" here, while
		// /api/me and /admin/stats — which have no envelope — call it "status".
		"status":           "ok",
		"version":          config.Version,
		"keyStatus":        customerKeyStatus(*entry),
		"requestsCount":    entry.RequestsCount,
		"tokensUsed":       entry.TokensUsed,
		"creditsUsed":      entry.CreditsUsed,
		"tokenLimit":       entry.TokenLimit,
		"creditLimit":      entry.CreditLimit,
		"tokensRemaining":  tokensRemaining(*entry),
		"creditsRemaining": creditsRemaining(*entry),
		"lastUsedAt":       entry.LastUsedAt,
		"uptime":           time.Now().Unix() - h.startTime,
	})
}

// handleCustomerMe GET /api/me — identity + quota view for the caller's key.
// Returns the masked key so customers can confirm which key they queried with,
// never the cleartext value.
func (h *Handler) handleCustomerMe(w http.ResponseWriter, r *http.Request) {
	entry := h.authenticateCustomerKey(w, r)
	if entry == nil {
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"id":               entry.ID,
		"name":             entry.Name,
		"keyMasked":        config.MaskApiKey(entry.Key),
		"status":           customerKeyStatus(*entry),
		"createdAt":        entry.CreatedAt,
		"lastUsedAt":       entry.LastUsedAt,
		"tokenLimit":       entry.TokenLimit,
		"creditLimit":      entry.CreditLimit,
		"tokensUsed":       entry.TokensUsed,
		"creditsUsed":      entry.CreditsUsed,
		"tokensRemaining":  tokensRemaining(*entry),
		"creditsRemaining": creditsRemaining(*entry),
		"requestsCount":    entry.RequestsCount,
	})
}

// customerLogView is the per-key request log payload. It intentionally omits
// AccountID and ApiKeyID: upstream pool account identifiers are internal
// operator data and the key ID is redundant (the caller already knows which
// key it authenticated with).
type customerLogView struct {
	Time                int64   `json:"time"`
	Endpoint            string  `json:"endpoint"`
	Model               string  `json:"model"`
	Status              string  `json:"status"`
	Error               string  `json:"error,omitempty"`
	ErrorType           string  `json:"errorType,omitempty"`
	Tokens              int     `json:"tokens"`
	InputTokens         int     `json:"inputTokens,omitempty"`
	OutputTokens        int     `json:"outputTokens,omitempty"`
	Credits             float64 `json:"credits"`
	Duration            int64   `json:"duration"`
	TTFT                int64   `json:"ttft,omitempty"`
	PayloadBytes        int     `json:"payloadBytes,omitempty"`
	HistoryBytes        int     `json:"historyBytes,omitempty"`
	ToolDefinitionBytes int     `json:"toolDefinitionBytes,omitempty"`
	ToolResultBytes     int     `json:"toolResultBytes,omitempty"`
}

// handleCustomerLogs GET /api/logs — recent request logs for the caller's key,
// newest first. Backed by the same in-memory ring buffer as the admin Request
// Logs page (last 500 requests across ALL keys), so older per-key history may
// have rotated out. Optional ?limit=N caps the response (default 100).
func (h *Handler) handleCustomerLogs(w http.ResponseWriter, r *http.Request) {
	entry := h.authenticateCustomerKey(w, r)
	if entry == nil {
		return
	}

	// Parse the optional limit; clamp to the ring buffer size to keep the
	// contract honest (we can never return more than the buffer holds).
	limit := 100
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = n
		}
	}
	if limit > requestLogsMaxSize {
		limit = requestLogsMaxSize
	}

	// getRequestLogs already returns newest-first copies; filter to this key.
	// Guard against entries with an empty ID (possible via hand-edited config):
	// an empty entry.ID would otherwise match every unattributed log line
	// (legacy/unauthenticated traffic) and leak other users' request metadata.
	logs := h.getRequestLogs()
	out := make([]customerLogView, 0, limit)
	for _, l := range logs {
		if entry.ID == "" || l.ApiKeyID == "" || l.ApiKeyID != entry.ID {
			continue
		}
		out = append(out, customerLogView{
			Time:                l.Time,
			Endpoint:            l.Endpoint,
			Model:               l.Model,
			Status:              l.Status,
			Error:               l.Error,
			ErrorType:           l.ErrorType,
			Tokens:              l.Tokens,
			InputTokens:         l.InputTokens,
			OutputTokens:        l.OutputTokens,
			Credits:             l.Credits,
			Duration:            l.Duration,
			TTFT:                l.TTFT,
			PayloadBytes:        l.PayloadBytes,
			HistoryBytes:        l.HistoryBytes,
			ToolDefinitionBytes: l.ToolDefinitionBytes,
			ToolResultBytes:     l.ToolResultBytes,
		})
		if len(out) >= limit {
			break
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"logs":  out,
		"count": len(out),
	})
}
