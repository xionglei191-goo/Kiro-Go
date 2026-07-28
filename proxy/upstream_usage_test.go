package proxy

import (
	"math"
	"testing"
)

func TestUpstreamUsageTrackerSeparatesKnownAndUnknownInput(t *testing.T) {
	tracker := newUpstreamUsageTracker()
	tracker.Record(KiroTokenUsage{
		InputTokens:              1000,
		OutputTokens:             20,
		UncachedInputTokens:      100,
		CacheReadInputTokens:     700,
		CacheCreationInputTokens: 200,
		InputBreakdownAvailable:  true,
	})
	tracker.Record(KiroTokenUsage{InputTokens: 400, OutputTokens: 10})

	got := tracker.Stats()
	if got.Calls != 2 || got.CallsWithBreakdown != 1 || got.CacheHitCalls != 1 {
		t.Fatalf("unexpected call counters: %+v", got)
	}
	if got.InputTokens != 1400 || got.OutputTokens != 30 || got.UnknownInputTokens != 400 {
		t.Fatalf("unexpected token totals: %+v", got)
	}
	if got.UncachedInputTokens != 100 ||
		got.CacheReadInputTokens != 700 ||
		got.CacheCreationInputTokens != 200 {
		t.Fatalf("unexpected breakdown totals: %+v", got)
	}
	if math.Abs(got.BreakdownCoverage-0.5) > 0.0001 {
		t.Fatalf("unexpected breakdown coverage: %f", got.BreakdownCoverage)
	}
	if math.Abs(got.CacheReadRatio-0.7) > 0.0001 {
		t.Fatalf("unexpected cache read ratio: %f", got.CacheReadRatio)
	}
}

func TestActualUpstreamPromptCacheUsageAggregatesCalls(t *testing.T) {
	first := kiroCallMetrics{usage: KiroTokenUsage{
		CacheReadInputTokens:     600,
		CacheCreationInputTokens: 200,
		InputBreakdownAvailable:  true,
	}}
	second := kiroCallMetrics{usage: KiroTokenUsage{
		CacheReadInputTokens:     50,
		CacheCreationInputTokens: 25,
		InputBreakdownAvailable:  true,
	}}

	got, ok := actualUpstreamPromptCacheUsage(1200, first, second)
	if !ok {
		t.Fatal("expected actual upstream cache usage")
	}
	if got.CacheReadInputTokens != 650 || got.CacheCreationInputTokens != 225 {
		t.Fatalf("unexpected aggregated usage: %+v", got)
	}
	if got.CacheCreation5mInputTokens != got.CacheCreationInputTokens {
		t.Fatalf("cache creation fallback must be classified as 5m: %+v", got)
	}
}

func TestActualUpstreamPromptCacheUsageClampsInvalidTotals(t *testing.T) {
	metric := kiroCallMetrics{usage: KiroTokenUsage{
		CacheReadInputTokens:     900,
		CacheCreationInputTokens: 300,
		InputBreakdownAvailable:  true,
	}}

	got, ok := actualUpstreamPromptCacheUsage(1000, metric)
	if !ok {
		t.Fatal("expected actual upstream cache usage")
	}
	if got.CacheReadInputTokens+got.CacheCreationInputTokens != 1000 {
		t.Fatalf("cache coverage must be clamped to total input: %+v", got)
	}
}

func TestEffectiveInputTokensPrefersExactUpstreamBreakdown(t *testing.T) {
	metrics := kiroCallMetrics{
		realInputTokens: 2000,
		inputTokens:     1000,
		usage: KiroTokenUsage{
			InputTokens:             1000,
			CacheReadInputTokens:    600,
			InputBreakdownAvailable: true,
		},
	}

	if got := metrics.effectiveInputTokens(50); got != 1000 {
		t.Fatalf("expected exact upstream total, got %d", got)
	}
}
