package proxy

import (
	"encoding/json"
	"sort"
)

const largeToolResultThresholdBytes = 64 * 1024

type requestPayloadMetrics struct {
	PayloadBytes        int
	HistoryBytes        int
	ToolDefinitionBytes int
	ToolResultBytes     int
	ToolDefinitionCount int
	ToolResultCount     int
}

type requestAuxiliaryMetrics struct {
	Purpose     string
	Model       string
	Outcome     string
	InputTokens int
	Credits     float64
	TTFT        int64
}

func measureKiroPayload(payload *KiroPayload) requestPayloadMetrics {
	if payload == nil {
		return requestPayloadMetrics{}
	}

	metrics := requestPayloadMetrics{
		PayloadBytes: jsonByteLen(payload),
	}
	if history := payload.ConversationState.History; len(history) > 0 {
		metrics.HistoryBytes = jsonByteLen(history)
	}

	context := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if context == nil {
		return metrics
	}
	if len(context.Tools) > 0 {
		metrics.ToolDefinitionBytes = jsonByteLen(context.Tools)
		metrics.ToolDefinitionCount = len(context.Tools)
	}
	if len(context.ToolResults) > 0 {
		metrics.ToolResultBytes = jsonByteLen(context.ToolResults)
		metrics.ToolResultCount = len(context.ToolResults)
	}
	return metrics
}

func jsonByteLen(value interface{}) int {
	data, err := json.Marshal(value)
	if err != nil {
		return 0
	}
	return len(data)
}

// RequestPerformanceStats summarizes the recent in-memory request log. These
// values describe proxy behavior only; they do not claim an upstream cache hit
// unless Kiro reports one through the separate upstreamCache statistics.
type RequestPerformanceStats struct {
	Samples                       int            `json:"samples"`
	SamplesWithTTFT               int            `json:"samplesWithTtft"`
	TTFTP50Ms                     int64          `json:"ttftP50Ms"`
	TTFTP95Ms                     int64          `json:"ttftP95Ms"`
	DurationP50Ms                 int64          `json:"durationP50Ms"`
	DurationP95Ms                 int64          `json:"durationP95Ms"`
	PayloadP50Bytes               int64          `json:"payloadP50Bytes"`
	PayloadP95Bytes               int64          `json:"payloadP95Bytes"`
	MaxPayloadBytes               int64          `json:"maxPayloadBytes"`
	ToolResultCalls               int            `json:"toolResultCalls"`
	LargeToolResultCalls          int            `json:"largeToolResultCalls"`
	LargeToolResultThresholdBytes int            `json:"largeToolResultThresholdBytes"`
	TotalToolResultBytes          int64          `json:"totalToolResultBytes"`
	TotalCredits                  float64        `json:"totalCredits"`
	AuxiliaryCalls                int            `json:"auxiliaryCalls"`
	AuxiliaryInputTokens          int64          `json:"auxiliaryInputTokens"`
	AuxiliaryCredits              float64        `json:"auxiliaryCredits"`
	AuxiliaryTTFTP50Ms            int64          `json:"auxiliaryTtftP50Ms"`
	AuxiliaryTTFTP95Ms            int64          `json:"auxiliaryTtftP95Ms"`
	ControllerCalls               int            `json:"controllerCalls"`
	ControllerModelCounts         map[string]int `json:"controllerModelCounts"`
	ControllerOutcomeCounts       map[string]int `json:"controllerOutcomeCounts"`
}

func requestPerformanceFromLogs(logs []RequestLog) RequestPerformanceStats {
	stats := RequestPerformanceStats{
		LargeToolResultThresholdBytes: largeToolResultThresholdBytes,
	}
	durations := make([]int64, 0, len(logs))
	ttfts := make([]int64, 0, len(logs))
	payloads := make([]int64, 0, len(logs))
	auxiliaryTTFTs := make([]int64, 0, len(logs))
	stats.ControllerModelCounts = make(map[string]int)
	stats.ControllerOutcomeCounts = make(map[string]int)

	for _, entry := range logs {
		if entry.Status != "success" || entry.PayloadBytes <= 0 {
			continue
		}

		stats.Samples++
		stats.TotalCredits += entry.Credits
		durations = append(durations, entry.Duration)
		payloads = append(payloads, int64(entry.PayloadBytes))
		if entry.TTFT > 0 {
			ttfts = append(ttfts, entry.TTFT)
		}
		if entry.ToolResultBytes > 0 {
			stats.ToolResultCalls++
			stats.TotalToolResultBytes += int64(entry.ToolResultBytes)
			if entry.ToolResultBytes >= largeToolResultThresholdBytes {
				stats.LargeToolResultCalls++
			}
		}
		if int64(entry.PayloadBytes) > stats.MaxPayloadBytes {
			stats.MaxPayloadBytes = int64(entry.PayloadBytes)
		}
		if entry.AuxiliaryPurpose != "" {
			stats.AuxiliaryCalls++
			stats.AuxiliaryInputTokens += int64(entry.AuxiliaryInputTokens)
			stats.AuxiliaryCredits += entry.AuxiliaryCredits
			if entry.AuxiliaryTTFT > 0 {
				auxiliaryTTFTs = append(auxiliaryTTFTs, entry.AuxiliaryTTFT)
			}
		}
		if entry.AuxiliaryPurpose == "controller" {
			stats.ControllerCalls++
			stats.ControllerModelCounts[entry.AuxiliaryModel]++
			stats.ControllerOutcomeCounts[entry.AuxiliaryOutcome]++
		}
	}

	stats.SamplesWithTTFT = len(ttfts)
	stats.TTFTP50Ms = nearestRankPercentile(ttfts, 50)
	stats.TTFTP95Ms = nearestRankPercentile(ttfts, 95)
	stats.DurationP50Ms = nearestRankPercentile(durations, 50)
	stats.DurationP95Ms = nearestRankPercentile(durations, 95)
	stats.PayloadP50Bytes = nearestRankPercentile(payloads, 50)
	stats.PayloadP95Bytes = nearestRankPercentile(payloads, 95)
	stats.AuxiliaryTTFTP50Ms = nearestRankPercentile(auxiliaryTTFTs, 50)
	stats.AuxiliaryTTFTP95Ms = nearestRankPercentile(auxiliaryTTFTs, 95)
	return stats
}

func nearestRankPercentile(values []int64, percentile int) int64 {
	if len(values) == 0 {
		return 0
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	rank := (percentile*len(values) + 99) / 100
	if rank < 1 {
		rank = 1
	}
	if rank > len(values) {
		rank = len(values)
	}
	return values[rank-1]
}
