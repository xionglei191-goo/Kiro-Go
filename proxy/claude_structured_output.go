package proxy

import (
	"encoding/json"
	"fmt"
	"kiro-go/config"
	"strings"

	"github.com/google/uuid"
)

const claudeStructuredOutputToolBase = "kiroGoStructuredOutput"

func claudeStructuredOutputSchema(req *ClaudeRequest) (interface{}, bool) {
	if req == nil || req.OutputConfig == nil || req.OutputConfig.Format == nil {
		return nil, false
	}
	format := req.OutputConfig.Format
	if !strings.EqualFold(strings.TrimSpace(format.Type), "json_schema") || format.Schema == nil {
		return nil, false
	}
	return ensureObjectSchema(format.Schema), true
}

func addClaudeStructuredOutputTool(tools []KiroToolWrapper, schema interface{}) ([]KiroToolWrapper, string) {
	name := chooseClaudeControllerToolName(tools, claudeStructuredOutputToolBase)
	var tool KiroToolWrapper
	tool.ToolSpecification.Name = name
	tool.ToolSpecification.Description = "Return the final response as structured JSON matching the required schema."
	tool.ToolSpecification.InputSchema.JSON = schema
	return append(tools, tool), name
}

func buildClaudeStructuredOutputPrompt(toolName string) string {
	return fmt.Sprintf(
		`The client requires a JSON response matching the provided schema. Invoke %q exactly once with the final response fields. Do not emit text, Markdown, or code fences; return the result only through that tool.`,
		toolName,
	)
}

func consumeClaudeStructuredOutputTool(payload *KiroPayload, toolUse KiroToolUse) (string, bool) {
	if payload == nil || payload.StructuredOutputToolName == "" || toolUse.Name != payload.StructuredOutputToolName {
		return "", false
	}
	encoded, err := json.Marshal(toolUse.Input)
	if err != nil {
		return "", true
	}
	return string(encoded), true
}

func buildClaudeStructuredOutputRetryPayload(payload *KiroPayload, previousContent string) *KiroPayload {
	if payload == nil || payload.StructuredOutputToolName == "" {
		return nil
	}

	structuredTool, ok := findKiroTool(payload, payload.StructuredOutputToolName)
	if !ok {
		return nil
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	var retry KiroPayload
	if err := json.Unmarshal(raw, &retry); err != nil {
		return nil
	}

	originalCurrent := retry.ConversationState.CurrentMessage.UserInputMessage
	historyUser := originalCurrent
	historyUser.UserInputMessageContext = nil
	retry.ConversationState.History = append(
		retry.ConversationState.History,
		KiroHistoryMessage{UserInputMessage: &historyUser},
	)
	if previousContent = strings.TrimSpace(previousContent); previousContent != "" {
		retry.ConversationState.History = append(
			retry.ConversationState.History,
			KiroHistoryMessage{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: previousContent}},
		)
	}
	retry.ConversationState.History = sanitizeKiroHistory(retry.ConversationState.History, nil)
	retry.ConversationState.AgentContinuationId = uuid.New().String()
	retry.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: fmt.Sprintf(
			`Your previous response did not satisfy the required structured-output protocol. Invoke %q now exactly once with arguments matching its schema. Emit no text or Markdown.`,
			payload.StructuredOutputToolName,
		),
		ModelID: originalCurrent.ModelID,
		Origin:  originalCurrent.Origin,
		UserInputMessageContext: &UserInputMessageContext{
			Tools: []KiroToolWrapper{structuredTool},
		},
	}
	retry.ToolNameMap = nil
	retry.ClaudeCodeAgent = false
	retry.ClaudeToolChoiceRequired = false
	retry.ControllerFinishToolName = ""
	retry.ControllerWaitToolName = ""
	retry.StructuredOutputToolName = payload.StructuredOutputToolName

	truncatePayloadToLimit(&retry, hasKiroSystemPriming(retry.ConversationState.History))
	return &retry
}

func callClaudeStructuredOutputRetry(
	account *config.Account,
	payload *KiroPayload,
	previousContent string,
	model string,
	metrics *kiroCallMetrics,
) (string, error) {
	retry := buildClaudeStructuredOutputRetryPayload(payload, previousContent)
	if retry == nil {
		return "", fmt.Errorf("could not build structured-output retry")
	}

	var textOutput strings.Builder
	var structuredOutput string
	err := CallKiroAPI(account, retry, metrics.callback(
		model,
		func(text string, isThinking bool) {
			if !isThinking {
				textOutput.WriteString(text)
			}
		},
		func(toolUse KiroToolUse) {
			if output, consumed := consumeClaudeStructuredOutputTool(retry, toolUse); consumed {
				structuredOutput = output
			}
		},
	))
	if err != nil {
		return "", err
	}
	if structuredOutput != "" {
		return structuredOutput, nil
	}
	if normalized, ok := normalizeClaudeStructuredJSON(textOutput.String()); ok {
		return normalized, nil
	}
	if coerced, ok := coerceClaudeGoalVerifierOutput(structuredOutputSchemaFromPayload(payload), textOutput.String()); ok {
		return coerced, nil
	}
	return "", fmt.Errorf("upstream did not return structured JSON")
}

func findKiroTool(payload *KiroPayload, name string) (KiroToolWrapper, bool) {
	for _, tool := range currentKiroTools(payload) {
		if tool.ToolSpecification.Name == name {
			return tool, true
		}
	}
	return KiroToolWrapper{}, false
}

func structuredOutputSchemaFromPayload(payload *KiroPayload) interface{} {
	tool, ok := findKiroTool(payload, payload.StructuredOutputToolName)
	if !ok {
		return nil
	}
	return tool.ToolSpecification.InputSchema.JSON
}

func normalizeClaudeStructuredJSON(content string) (string, bool) {
	trimmed := strings.TrimSpace(content)
	candidates := []string{trimmed}
	if strings.HasPrefix(trimmed, "```") {
		if newline := strings.IndexByte(trimmed, '\n'); newline >= 0 {
			fenced := strings.TrimSpace(trimmed[newline+1:])
			fenced = strings.TrimSpace(strings.TrimSuffix(fenced, "```"))
			candidates = append(candidates, fenced)
		}
	}
	if start, end := strings.Index(trimmed, "{"), strings.LastIndex(trimmed, "}"); start >= 0 && end > start {
		candidates = append(candidates, trimmed[start:end+1])
	}

	for _, candidate := range candidates {
		var object map[string]interface{}
		if err := json.Unmarshal([]byte(candidate), &object); err != nil {
			continue
		}
		encoded, err := json.Marshal(object)
		if err == nil {
			return string(encoded), true
		}
	}
	return "", false
}

func coerceClaudeGoalVerifierOutput(schema interface{}, content string) (string, bool) {
	schemaMap, ok := schema.(map[string]interface{})
	if !ok {
		return "", false
	}
	properties, ok := schemaMap["properties"].(map[string]interface{})
	if !ok || !schemaPropertyHasType(properties["ok"], "boolean") || !schemaPropertyHasType(properties["reason"], "string") {
		return "", false
	}

	reason := strings.TrimSpace(content)
	if reason == "" {
		return "", false
	}
	lower := strings.ToLower(strings.TrimLeft(reason, "*#_ \t\r\n"))
	met := false
	switch {
	case strings.HasPrefix(lower, "no"),
		strings.Contains(lower, "not been satisfied"),
		strings.Contains(lower, "condition has not"),
		strings.Contains(lower, "condition is not met"),
		strings.Contains(lower, "does not satisfy"),
		strings.Contains(reason, "未满足"),
		strings.Contains(reason, "未完成"):
		met = false
	case strings.HasPrefix(lower, "yes"),
		strings.Contains(lower, "condition has been satisfied"),
		strings.Contains(lower, "condition is met"),
		strings.Contains(reason, "已满足"):
		met = true
	default:
		return "", false
	}

	encoded, err := json.Marshal(map[string]interface{}{"ok": met, "reason": reason})
	if err != nil {
		return "", false
	}
	return string(encoded), true
}

func schemaPropertyHasType(property interface{}, expected string) bool {
	propertyMap, ok := property.(map[string]interface{})
	if !ok {
		return false
	}
	actual, _ := propertyMap["type"].(string)
	return actual == expected
}
