package proxy

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/google/uuid"
)

const claudeToolAutoContinuationPrompt = `Continue the task now. Your previous response stated an intended action but did not execute it. Invoke the next required provided tool in this response. Do not repeat the plan, describe the action, or return a final answer before the tool call.`

const maxClaudeToolIntentRunes = 1200

var (
	claudeEnglishToolIntentPattern     = regexp.MustCompile(`(?im)(?:^|[\n.!]\s*)(?:[-*]\s*)?(?:(?:okay|sure|all right)[,:]?\s+)?(?:(?:i(?:'ll| will| am going to|'m going to| need to| should)|we(?:'ll| will)|let me|let's|(?:next|first|now),?\s+i(?:'ll| will))\s+)(?:quickly\s+|now\s+)?(?:inspect|check|read|search|look(?:\s+at)?|run|execute|edit|modify|update|implement|fix|test|build|deploy|restart|verify|open|fetch|investigate|analy[sz]e|review|create|add|remove|apply|continue|proceed|use)\b`)
	claudeEnglishProgressIntentPattern = regexp.MustCompile(`(?im)(?:^|\n)\s*(?:[-*]\s*)?(?:checking|inspecting|reading|searching|running|executing|editing|updating|implementing|testing|building|deploying|restarting|verifying|opening|fetching|investigating|reviewing)\b`)
	claudeChineseToolIntentPattern     = regexp.MustCompile(`(?m)(?:^|[\n。！？.!]\s*)(?:[-*]\s*)?(?:好(?:的)?[，,]\s*)?(?:(?:我(?:先|现在|接下来|下一步)?(?:来|会|将|要|需要|准备|继续)?|让我|(?:现在|接下来|下一步|首先|先)(?:我)?(?:来|会|将|要|需要|准备|继续)?)\s*)(?:先|马上|立即|继续|重新|再)?\s*(?:检查|查看|读取|搜索|查找|运行|执行|修改|编辑|更新|实现|修复|测试|构建|编译|部署|重启|验证|打开|获取|定位|分析|审查|创建|添加|删除|应用|处理)`)
	claudeIntentTailSentenceBoundary   = regexp.MustCompile(`[.!?。！？]\s+`)
)

type kiroCallMetrics struct {
	inputTokens     int
	realInputTokens int
	credits         float64
}

func (m *kiroCallMetrics) callback(model string, onText func(string, bool), onToolUse func(KiroToolUse)) *KiroStreamCallback {
	return &KiroStreamCallback{
		OnText:    onText,
		OnToolUse: onToolUse,
		OnComplete: func(inputTokens, _ int) {
			m.inputTokens = inputTokens
		},
		OnCredits: func(credits float64) {
			m.credits += credits
		},
		OnContextUsage: func(percentage float64) {
			m.realInputTokens = int(percentage * float64(getContextWindowSize(model)) / 100.0)
		},
	}
}

func (m kiroCallMetrics) effectiveInputTokens(fallback int) int {
	if m.realInputTokens > 0 {
		return m.realInputTokens
	}
	if m.inputTokens > 0 {
		return m.inputTokens
	}
	return fallback
}

func shouldAutoContinueClaudeToolCall(payload *KiroPayload, content string, toolUses []KiroToolUse) bool {
	return claudeToolAutoContinueDecision(payload, content, toolUses) == "action_intent"
}

func claudeToolAutoContinueDecision(payload *KiroPayload, content string, toolUses []KiroToolUse) string {
	if payload == nil {
		return "no_payload"
	}
	if len(toolUses) > 0 {
		return "tool_already_used"
	}
	if len(currentKiroTools(payload)) == 0 {
		return "no_tools"
	}
	content = strings.TrimSpace(content)
	if content == "" {
		return "empty"
	}
	if len([]rune(content)) > maxClaudeToolIntentRunes {
		return "long_response"
	}
	if strings.HasSuffix(content, "?") || strings.HasSuffix(content, "？") {
		return "question"
	}

	if hasTerminalClaudeToolIntent(claudeEnglishToolIntentPattern, content) ||
		hasTerminalClaudeToolIntent(claudeEnglishProgressIntentPattern, content) ||
		hasTerminalClaudeToolIntent(claudeChineseToolIntentPattern, content) {
		return "action_intent"
	}
	return "no_action_intent"
}

func hasTerminalClaudeToolIntent(pattern *regexp.Regexp, content string) bool {
	matches := pattern.FindAllStringIndex(content, -1)
	for i := len(matches) - 1; i >= 0; i-- {
		tail := strings.TrimSpace(content[matches[i][1]:])
		tail = strings.TrimSpace(strings.TrimRight(tail, ".!。！"))
		if !claudeIntentTailSentenceBoundary.MatchString(tail) {
			return true
		}
	}
	return false
}

func currentKiroTools(payload *KiroPayload) []KiroToolWrapper {
	if payload == nil {
		return nil
	}
	context := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if context == nil {
		return nil
	}
	return context.Tools
}

func buildClaudeToolAutoContinuationPayload(payload *KiroPayload, assistantContent string) *KiroPayload {
	if payload == nil || strings.TrimSpace(assistantContent) == "" {
		return nil
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	var continuation KiroPayload
	if err := json.Unmarshal(raw, &continuation); err != nil {
		return nil
	}
	continuation.ToolNameMap = make(map[string]string, len(payload.ToolNameMap))
	for sanitized, original := range payload.ToolNameMap {
		continuation.ToolNameMap[sanitized] = original
	}

	hasPriming := hasKiroSystemPriming(continuation.ConversationState.History)
	originalCurrent := continuation.ConversationState.CurrentMessage.UserInputMessage
	var tools []KiroToolWrapper
	if originalCurrent.UserInputMessageContext != nil {
		tools = append(tools, originalCurrent.UserInputMessageContext.Tools...)
	}

	continuation.ConversationState.History = append(
		continuation.ConversationState.History,
		KiroHistoryMessage{UserInputMessage: &originalCurrent},
		KiroHistoryMessage{AssistantResponseMessage: &KiroAssistantResponseMessage{
			Content: strings.TrimSpace(assistantContent),
		}},
	)
	continuation.ConversationState.History = sanitizeKiroHistory(continuation.ConversationState.History, nil)
	continuation.ConversationState.AgentContinuationId = uuid.New().String()
	continuation.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: claudeToolAutoContinuationPrompt,
		ModelID: originalCurrent.ModelID,
		Origin:  originalCurrent.Origin,
		UserInputMessageContext: &UserInputMessageContext{
			Tools: tools,
		},
	}

	truncatePayloadToLimit(&continuation, hasPriming)
	return &continuation
}

func hasKiroSystemPriming(history []KiroHistoryMessage) bool {
	if len(history) < 2 || history[0].UserInputMessage == nil || history[1].AssistantResponseMessage == nil {
		return false
	}
	return history[1].AssistantResponseMessage.Content == "I will follow these instructions."
}

func claudeConversationLogID(payload *KiroPayload) string {
	if payload == nil {
		return ""
	}
	id := payload.ConversationState.ConversationID
	if len(id) > 8 {
		return id[:8]
	}
	return id
}
