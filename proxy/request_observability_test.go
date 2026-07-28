package proxy

import (
	"testing"
	"time"
)

func TestMeasureKiroPayloadBreaksOutCurrentContext(t *testing.T) {
	payload := &KiroPayload{}
	payload.ConversationState.History = []KiroHistoryMessage{{
		AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "previous response"},
	}}
	context := &UserInputMessageContext{
		Tools:       make([]KiroToolWrapper, 2),
		ToolResults: make([]KiroToolResult, 1),
	}
	context.Tools[0].ToolSpecification.Name = "read_file"
	context.Tools[1].ToolSpecification.Name = "run_command"
	context.ToolResults[0] = KiroToolResult{
		ToolUseID: "tool-1",
		Status:    "success",
		Content:   []KiroResultContent{{Text: "build passed"}},
	}
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext = context

	got := measureKiroPayload(payload)
	if got.PayloadBytes != jsonByteLen(payload) {
		t.Fatalf("payload bytes = %d, want %d", got.PayloadBytes, jsonByteLen(payload))
	}
	if got.HistoryBytes != jsonByteLen(payload.ConversationState.History) {
		t.Fatalf("history bytes = %d, want %d", got.HistoryBytes, jsonByteLen(payload.ConversationState.History))
	}
	if got.ToolDefinitionBytes != jsonByteLen(context.Tools) || got.ToolDefinitionCount != 2 {
		t.Fatalf("unexpected tool definition metrics: %+v", got)
	}
	if got.ToolResultBytes != jsonByteLen(context.ToolResults) || got.ToolResultCount != 1 {
		t.Fatalf("unexpected tool result metrics: %+v", got)
	}
}

func TestRequestPerformanceFromLogsUsesRecentObservedSuccesses(t *testing.T) {
	logs := []RequestLog{
		{Status: "success", PayloadBytes: 100, Duration: 1000, TTFT: 100, Credits: 0.1},
		{
			Status: "success", PayloadBytes: 200, Duration: 2000, TTFT: 200, Credits: 0.2, ToolResultBytes: 1024,
			AuxiliaryPurpose: "controller", AuxiliaryModel: "claude-haiku-4.5", AuxiliaryOutcome: "continue",
			AuxiliaryInputTokens: 1200, AuxiliaryCredits: 0.05, AuxiliaryTTFT: 80,
		},
		{
			Status: "success", PayloadBytes: 300, Duration: 3000, TTFT: 300, Credits: 0.3, ToolResultBytes: largeToolResultThresholdBytes,
			AuxiliaryPurpose: "controller", AuxiliaryModel: "claude-haiku-4.5", AuxiliaryOutcome: "finish",
			AuxiliaryInputTokens: 1300, AuxiliaryCredits: 0.06, AuxiliaryTTFT: 100,
		},
		{Status: "success", PayloadBytes: 400, Duration: 4000, TTFT: 400, Credits: 0.4},
		{Status: "error", PayloadBytes: 9999, Duration: 9999, TTFT: 9999},
		{Status: "success", Duration: 5000, TTFT: 500},
	}

	got := requestPerformanceFromLogs(logs)
	if got.Samples != 4 || got.SamplesWithTTFT != 4 {
		t.Fatalf("unexpected sample counts: %+v", got)
	}
	if got.TTFTP50Ms != 200 || got.TTFTP95Ms != 400 {
		t.Fatalf("unexpected TTFT percentiles: %+v", got)
	}
	if got.DurationP50Ms != 2000 || got.DurationP95Ms != 4000 {
		t.Fatalf("unexpected duration percentiles: %+v", got)
	}
	if got.PayloadP50Bytes != 200 || got.PayloadP95Bytes != 400 || got.MaxPayloadBytes != 400 {
		t.Fatalf("unexpected payload percentiles: %+v", got)
	}
	if got.ToolResultCalls != 2 || got.LargeToolResultCalls != 1 {
		t.Fatalf("unexpected tool result counts: %+v", got)
	}
	if got.TotalToolResultBytes != 1024+largeToolResultThresholdBytes {
		t.Fatalf("unexpected total tool result bytes: %+v", got)
	}
	if got.TotalCredits < 0.999 || got.TotalCredits > 1.001 {
		t.Fatalf("unexpected credits total: %+v", got)
	}
	if got.AuxiliaryCalls != 2 || got.ControllerCalls != 2 ||
		got.AuxiliaryInputTokens != 2500 ||
		got.AuxiliaryCredits < 0.109 || got.AuxiliaryCredits > 0.111 {
		t.Fatalf("unexpected auxiliary totals: %+v", got)
	}
	if got.AuxiliaryTTFTP50Ms != 80 || got.AuxiliaryTTFTP95Ms != 100 {
		t.Fatalf("unexpected auxiliary TTFT percentiles: %+v", got)
	}
	if got.ControllerModelCounts["claude-haiku-4.5"] != 2 ||
		got.ControllerOutcomeCounts["continue"] != 1 ||
		got.ControllerOutcomeCounts["finish"] != 1 {
		t.Fatalf("unexpected controller breakdown: %+v", got)
	}
}

func TestKiroCallMetricsRecordsFirstContentOnce(t *testing.T) {
	requestStartedAt := time.Now().Add(-50 * time.Millisecond)
	metrics := kiroCallMetrics{upstreamUsage: newUpstreamUsageTracker()}
	textCalls := 0
	toolCalls := 0
	callback := metrics.callback(
		"claude-opus-4.1",
		func(string, bool) { textCalls++ },
		func(KiroToolUse) { toolCalls++ },
	)

	callback.OnText("hello", false)
	first := metrics.firstContentAt
	callback.OnToolUse(KiroToolUse{Name: "read_file"})

	if textCalls != 1 || toolCalls != 1 {
		t.Fatalf("callbacks were not forwarded: text=%d tool=%d", textCalls, toolCalls)
	}
	if first.IsZero() || !metrics.firstContentAt.Equal(first) {
		t.Fatalf("first content timestamp changed: first=%v current=%v", first, metrics.firstContentAt)
	}
	if got := metrics.ttftFrom(requestStartedAt); got < 40 {
		t.Fatalf("TTFT = %dms, want at least 40ms", got)
	}
}
