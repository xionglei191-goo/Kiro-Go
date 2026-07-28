package proxy

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

const (
	claudeControllerFinishToolBase = "kiroGoFinishTask"
	claudeControllerWaitToolBase   = "kiroGoWaitForUser"
)

type claudeControllerOutcome string

const (
	claudeControllerNotApplicable claudeControllerOutcome = "not_applicable"
	claudeControllerContinue      claudeControllerOutcome = "continue"
	claudeControllerFinish        claudeControllerOutcome = "finish"
	claudeControllerWaitForUser   claudeControllerOutcome = "wait_for_user"
	claudeControllerUndecided     claudeControllerOutcome = "undecided"
	claudeControllerUpstreamError claudeControllerOutcome = "upstream_error"
)

type kiroCallMetrics struct {
	inputTokens     int
	realInputTokens int
	credits         float64
	usage           KiroTokenUsage
	upstreamUsage   *upstreamUsageTracker
	firstContentAt  time.Time
}

func (m *kiroCallMetrics) callback(model string, onText func(string, bool), onToolUse func(KiroToolUse)) *KiroStreamCallback {
	return &KiroStreamCallback{
		OnText: func(text string, isThinking bool) {
			m.markFirstContent()
			if onText != nil {
				onText(text, isThinking)
			}
		},
		OnToolUse: func(toolUse KiroToolUse) {
			m.markFirstContent()
			if onToolUse != nil {
				onToolUse(toolUse)
			}
		},
		OnComplete: func(inputTokens, _ int) {
			m.inputTokens = inputTokens
		},
		OnUsage: func(usage KiroTokenUsage) {
			m.usage = usage
			m.upstreamUsage.Record(usage)
		},
		OnCredits: func(credits float64) {
			m.credits += credits
		},
		OnContextUsage: func(percentage float64) {
			m.realInputTokens = int(percentage * float64(getContextWindowSize(model)) / 100.0)
		},
	}
}

func (m *kiroCallMetrics) markFirstContent() {
	if m.firstContentAt.IsZero() {
		m.firstContentAt = time.Now()
	}
}

func (m kiroCallMetrics) ttftFrom(requestStartedAt time.Time) int64 {
	if m.firstContentAt.IsZero() || requestStartedAt.IsZero() {
		return 0
	}
	return m.firstContentAt.Sub(requestStartedAt).Milliseconds()
}

func (m kiroCallMetrics) effectiveInputTokens(fallback int) int {
	if m.usage.InputBreakdownAvailable && m.usage.InputTokens > 0 {
		return m.usage.InputTokens
	}
	if m.realInputTokens > 0 {
		return m.realInputTokens
	}
	if m.inputTokens > 0 {
		return m.inputTokens
	}
	return fallback
}

func shouldRunClaudeToolController(payload *KiroPayload, toolUses []KiroToolUse) bool {
	return payload != nil &&
		payload.ClaudeCodeAgent &&
		payload.StructuredOutputToolName == "" &&
		len(toolUses) == 0 &&
		len(currentKiroRealTools(payload)) > 0
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

func currentKiroRealTools(payload *KiroPayload) []KiroToolWrapper {
	tools := currentKiroTools(payload)
	if len(tools) == 0 {
		return nil
	}
	realTools := make([]KiroToolWrapper, 0, len(tools))
	for _, tool := range tools {
		name := tool.ToolSpecification.Name
		if name == payload.ControllerFinishToolName ||
			name == payload.ControllerWaitToolName ||
			name == payload.StructuredOutputToolName {
			continue
		}
		realTools = append(realTools, tool)
	}
	return realTools
}

func buildClaudeToolControllerPayload(payload *KiroPayload, assistantContent string) *KiroPayload {
	if payload == nil || !payload.ClaudeCodeAgent {
		return nil
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	var controller KiroPayload
	if err := json.Unmarshal(raw, &controller); err != nil {
		return nil
	}
	controller.ClaudeCodeAgent = payload.ClaudeCodeAgent
	controller.ClaudeToolChoiceRequired = payload.ClaudeToolChoiceRequired
	controller.StructuredOutputToolName = payload.StructuredOutputToolName
	controller.ToolNameMap = make(map[string]string, len(payload.ToolNameMap))
	for sanitized, original := range payload.ToolNameMap {
		controller.ToolNameMap[sanitized] = original
	}

	hasPriming := hasKiroSystemPriming(controller.ConversationState.History)
	originalCurrent := controller.ConversationState.CurrentMessage.UserInputMessage
	realTools := currentKiroRealTools(payload)
	if len(realTools) == 0 {
		return nil
	}

	controllerTools := append([]KiroToolWrapper{}, realTools...)
	if !payload.ClaudeToolChoiceRequired {
		finishName := chooseClaudeControllerToolName(controllerTools, claudeControllerFinishToolBase)
		controllerTools = append(controllerTools, newClaudeControllerTool(
			finishName,
			"Signal that the user's task is fully complete. Use only when no implementation, command, monitoring, verification, cleanup, or other autonomous work remains.",
		))
		waitName := chooseClaudeControllerToolName(controllerTools, claudeControllerWaitToolBase)
		controllerTools = append(controllerTools, newClaudeControllerTool(
			waitName,
			"Signal that progress is blocked on explicit user input, approval, credentials, or a decision that cannot be obtained with the available tools.",
		))
		controller.ControllerFinishToolName = finishName
		controller.ControllerWaitToolName = waitName
	}

	assistantContent = strings.TrimSpace(assistantContent)
	if assistantContent == "" {
		assistantContent = "(The previous assistant turn returned no text and no tool call.)"
	}
	controller.ConversationState.History = append(
		controller.ConversationState.History,
		KiroHistoryMessage{UserInputMessage: &originalCurrent},
		KiroHistoryMessage{AssistantResponseMessage: &KiroAssistantResponseMessage{
			Content: assistantContent,
		}},
	)
	controller.ConversationState.History = sanitizeKiroHistory(controller.ConversationState.History, nil)
	controller.ConversationState.AgentContinuationId = uuid.New().String()
	controller.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: buildClaudeControllerPrompt(
			controller.ControllerFinishToolName,
			controller.ControllerWaitToolName,
			payload.ClaudeToolChoiceRequired,
		),
		ModelID: originalCurrent.ModelID,
		Origin:  originalCurrent.Origin,
		UserInputMessageContext: &UserInputMessageContext{
			Tools: controllerTools,
		},
	}

	truncatePayloadToLimit(&controller, hasPriming)
	return &controller
}

func buildClaudeControllerPrompt(finishName, waitName string, toolChoiceRequired bool) string {
	if toolChoiceRequired {
		return `Act as the execution controller for the current Claude Code task.
Review the original user task, conversation history, tool results, and the previous assistant response.
The client requires a real tool call in this turn. Invoke one or more appropriate provided tools now without emitting text.
Never describe what you intend to do; emit the selected tool call now.`
	}
	return fmt.Sprintf(`Act as the execution controller for the current Claude Code task.
Review the original user task, conversation history, tool results, and the previous assistant response.
Make exactly one structured decision now, without emitting text:
- If autonomous work remains, invoke one or more appropriate real tools now.
- If the task is fully complete and no command, build, deployment, test, monitoring, verification, or cleanup remains, invoke %q.
- If progress genuinely requires explicit user input, approval, credentials, or a decision that no available tool can obtain, invoke %q.
Never signal completion merely because a command or background process is still running. Never describe what you intend to do; emit the selected tool call now.`, finishName, waitName)
}

func chooseClaudeControllerToolName(tools []KiroToolWrapper, base string) string {
	used := make(map[string]bool, len(tools))
	for _, tool := range tools {
		used[tool.ToolSpecification.Name] = true
	}
	if !used[base] {
		return base
	}
	for suffix := 2; ; suffix++ {
		candidate := fmt.Sprintf("%s%d", base, suffix)
		if !used[candidate] {
			return candidate
		}
	}
}

func newClaudeControllerTool(name, description string) KiroToolWrapper {
	var tool KiroToolWrapper
	tool.ToolSpecification.Name = name
	tool.ToolSpecification.Description = description
	tool.ToolSpecification.InputSchema.JSON = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"reason": map[string]interface{}{
				"type":        "string",
				"description": "A concise reason for this controller decision.",
			},
		},
		"required": []string{"reason"},
	}
	return tool
}

func splitClaudeControllerToolUses(payload *KiroPayload, toolUses []KiroToolUse) ([]KiroToolUse, claudeControllerOutcome) {
	if payload == nil {
		return nil, claudeControllerUndecided
	}

	realTools := make([]KiroToolUse, 0, len(toolUses))
	hasFinish := false
	hasWait := false
	for _, toolUse := range toolUses {
		switch toolUse.Name {
		case payload.ControllerFinishToolName:
			hasFinish = true
		case payload.ControllerWaitToolName:
			hasWait = true
		default:
			realTools = append(realTools, toolUse)
		}
	}

	if len(realTools) > 0 {
		return realTools, claudeControllerContinue
	}
	if hasWait {
		return nil, claudeControllerWaitForUser
	}
	if hasFinish {
		return nil, claudeControllerFinish
	}
	return nil, claudeControllerUndecided
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
