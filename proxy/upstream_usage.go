package proxy

import "sync/atomic"

// UpstreamCacheStats reports only usage observed in Kiro upstream responses.
// It is intentionally separate from PromptCacheStats, which predicts Anthropic
// prompt-cache behavior from request prefixes.
type UpstreamCacheStats struct {
	Calls                    int64   `json:"calls"`
	CallsWithBreakdown       int64   `json:"callsWithBreakdown"`
	CacheHitCalls            int64   `json:"cacheHitCalls"`
	InputTokens              int64   `json:"inputTokens"`
	OutputTokens             int64   `json:"outputTokens"`
	UncachedInputTokens      int64   `json:"uncachedInputTokens"`
	CacheReadInputTokens     int64   `json:"cacheReadInputTokens"`
	CacheCreationInputTokens int64   `json:"cacheCreationInputTokens"`
	UnknownInputTokens       int64   `json:"unknownInputTokens"`
	BreakdownCoverage        float64 `json:"breakdownCoverage"`
	CacheReadRatio           float64 `json:"cacheReadRatio"`
}

type upstreamUsageTracker struct {
	calls                    int64
	callsWithBreakdown       int64
	cacheHitCalls            int64
	inputTokens              int64
	outputTokens             int64
	uncachedInputTokens      int64
	cacheReadInputTokens     int64
	cacheCreationInputTokens int64
	unknownInputTokens       int64
}

func newUpstreamUsageTracker() *upstreamUsageTracker {
	return &upstreamUsageTracker{}
}

func (t *upstreamUsageTracker) Record(usage KiroTokenUsage) {
	if t == nil {
		return
	}

	atomic.AddInt64(&t.calls, 1)
	atomic.AddInt64(&t.inputTokens, int64(maxInt(usage.InputTokens, 0)))
	atomic.AddInt64(&t.outputTokens, int64(maxInt(usage.OutputTokens, 0)))
	if !usage.InputBreakdownAvailable {
		atomic.AddInt64(&t.unknownInputTokens, int64(maxInt(usage.InputTokens, 0)))
		return
	}

	atomic.AddInt64(&t.callsWithBreakdown, 1)
	atomic.AddInt64(&t.uncachedInputTokens, int64(maxInt(usage.UncachedInputTokens, 0)))
	atomic.AddInt64(&t.cacheReadInputTokens, int64(maxInt(usage.CacheReadInputTokens, 0)))
	atomic.AddInt64(&t.cacheCreationInputTokens, int64(maxInt(usage.CacheCreationInputTokens, 0)))
	if usage.CacheReadInputTokens > 0 {
		atomic.AddInt64(&t.cacheHitCalls, 1)
	}
}

func (t *upstreamUsageTracker) Stats() UpstreamCacheStats {
	if t == nil {
		return UpstreamCacheStats{}
	}

	stats := UpstreamCacheStats{
		Calls:                    atomic.LoadInt64(&t.calls),
		CallsWithBreakdown:       atomic.LoadInt64(&t.callsWithBreakdown),
		CacheHitCalls:            atomic.LoadInt64(&t.cacheHitCalls),
		InputTokens:              atomic.LoadInt64(&t.inputTokens),
		OutputTokens:             atomic.LoadInt64(&t.outputTokens),
		UncachedInputTokens:      atomic.LoadInt64(&t.uncachedInputTokens),
		CacheReadInputTokens:     atomic.LoadInt64(&t.cacheReadInputTokens),
		CacheCreationInputTokens: atomic.LoadInt64(&t.cacheCreationInputTokens),
		UnknownInputTokens:       atomic.LoadInt64(&t.unknownInputTokens),
	}
	if stats.Calls > 0 {
		stats.BreakdownCoverage = float64(stats.CallsWithBreakdown) / float64(stats.Calls)
	}
	knownInput := stats.UncachedInputTokens +
		stats.CacheReadInputTokens +
		stats.CacheCreationInputTokens
	if knownInput > 0 {
		stats.CacheReadRatio = float64(stats.CacheReadInputTokens) / float64(knownInput)
	}
	return stats
}

func actualUpstreamPromptCacheUsage(totalInputTokens int, metrics ...kiroCallMetrics) (promptCacheUsage, bool) {
	cacheRead := 0
	cacheCreation := 0
	hasBreakdown := false
	for _, metric := range metrics {
		if !metric.usage.InputBreakdownAvailable {
			continue
		}
		hasBreakdown = true
		cacheRead += maxInt(metric.usage.CacheReadInputTokens, 0)
		cacheCreation += maxInt(metric.usage.CacheCreationInputTokens, 0)
	}
	if !hasBreakdown {
		return promptCacheUsage{}, false
	}

	totalInputTokens = maxInt(totalInputTokens, 0)
	covered := cacheRead + cacheCreation
	if covered > totalInputTokens && covered > 0 {
		cacheRead = int(float64(cacheRead)*float64(totalInputTokens)/float64(covered) + 0.5)
		cacheRead = minInt(cacheRead, totalInputTokens)
		cacheCreation = totalInputTokens - cacheRead
	}

	return promptCacheUsage{
		CacheCreationInputTokens:   cacheCreation,
		CacheReadInputTokens:       cacheRead,
		CacheCreation5mInputTokens: cacheCreation,
	}, true
}
