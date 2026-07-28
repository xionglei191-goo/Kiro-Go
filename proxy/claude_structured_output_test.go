package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestClaudeToKiroBridgesJSONSchemaOutputWithInternalTool(t *testing.T) {
	payload := claudeStructuredOutputTestPayload()
	if payload.StructuredOutputToolName == "" {
		t.Fatal("structured-output tool name was not assigned")
	}
	if payload.ClaudeCodeAgent {
		t.Fatal("an internal structured-output tool must not enable the Claude Code controller")
	}
	if len(currentKiroRealTools(payload)) != 0 {
		t.Fatalf("internal structured-output tool leaked into real tools: %#v", currentKiroRealTools(payload))
	}

	current := payload.ConversationState.CurrentMessage.UserInputMessage
	if !strings.Contains(current.Content, payload.StructuredOutputToolName) ||
		!strings.Contains(current.Content, "Do not emit text") {
		t.Fatalf("structured-output instruction is missing: %q", current.Content)
	}
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.Tools) != 1 {
		t.Fatalf("expected exactly one internal structured-output tool: %#v", current.UserInputMessageContext)
	}
	tool := current.UserInputMessageContext.Tools[0]
	if tool.ToolSpecification.Name != payload.StructuredOutputToolName {
		t.Fatalf("unexpected structured-output tool: %#v", tool)
	}
	schema, ok := tool.ToolSpecification.InputSchema.JSON.(map[string]interface{})
	if !ok || schema["type"] != "object" {
		t.Fatalf("structured-output schema was not preserved: %#v", tool.ToolSpecification.InputSchema.JSON)
	}
}

func TestClaudeHandlersReturnStructuredOutputAsJSONText(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, request KiroPayload) []byte {
				if call != 1 {
					return nil
				}
				return awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
					"toolUseId":   "toolu_structured",
					"name":        structuredToolNameFromRequest(t, request),
					"input":       `{"ok":false,"reason":"work remains"}`,
					"stop":        true,
					"inputTokens": 50,
				})
			}, func(payload *KiroPayload) {
				*payload = *claudeStructuredOutputTestPayload()
			})

			if len(requests) != 1 {
				t.Fatalf("structured output should complete in one upstream request, got %d", len(requests))
			}
			assertStructuredClaudeResponse(t, recorder.Body.String(), stream, `{"ok":false,"reason":"work remains"}`, 50)
		})
	}
}

func TestClaudeHandlersRetryStructuredOutputWithoutLeakingInvalidText(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, request KiroPayload) []byte {
				switch call {
				case 1:
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content":     "**No, the condition has not been satisfied.**",
						"inputTokens": 40,
					})
				case 2:
					return awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
						"toolUseId":   "toolu_structured_retry",
						"name":        structuredToolNameFromRequest(t, request),
						"input":       `{"ok":false,"reason":"tests are still running"}`,
						"stop":        true,
						"inputTokens": 60,
					})
				default:
					return nil
				}
			}, func(payload *KiroPayload) {
				*payload = *claudeStructuredOutputTestPayload()
			})

			if len(requests) != 2 {
				t.Fatalf("expected one structured-output retry, got %d requests", len(requests))
			}
			retryCurrent := requests[1].ConversationState.CurrentMessage.UserInputMessage
			if retryCurrent.UserInputMessageContext == nil || len(retryCurrent.UserInputMessageContext.Tools) != 1 ||
				!strings.Contains(retryCurrent.Content, "Emit no text or Markdown") {
				t.Fatalf("unexpected structured-output retry request: %#v", retryCurrent)
			}
			if strings.Contains(recorder.Body.String(), "condition has not been satisfied") {
				t.Fatalf("invalid first response leaked to the client: %s", recorder.Body.String())
			}
			assertStructuredClaudeResponse(t, recorder.Body.String(), stream, `{"ok":false,"reason":"tests are still running"}`, 100)
		})
	}
}

func TestNormalizeAndCoerceClaudeGoalVerifierOutput(t *testing.T) {
	fenced := "```json\n{\"ok\":false,\"reason\":\"pending\"}\n```"
	normalized, ok := normalizeClaudeStructuredJSON(fenced)
	if !ok || normalized != `{"ok":false,"reason":"pending"}` {
		t.Fatalf("fenced JSON was not normalized: %q, %v", normalized, ok)
	}

	schema, _ := claudeStructuredOutputSchema(&ClaudeRequest{
		OutputConfig: &ClaudeOutputConfig{
			Format: &ClaudeOutputFormat{
				Type: "json_schema",
				Schema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"ok":     map[string]interface{}{"type": "boolean"},
						"reason": map[string]interface{}{"type": "string"},
					},
				},
			},
		},
	})
	coerced, ok := coerceClaudeGoalVerifierOutput(schema, "**No, the condition has not been satisfied.**")
	if !ok {
		t.Fatal("goal verifier prose was not coerced")
	}
	var result struct {
		OK     bool   `json:"ok"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(coerced), &result); err != nil {
		t.Fatalf("decode coerced result: %v", err)
	}
	if result.OK || result.Reason == "" {
		t.Fatalf("unexpected coerced result: %#v", result)
	}
}

func assertStructuredClaudeResponse(t *testing.T, body string, stream bool, expected string, inputTokens int) {
	t.Helper()
	if stream {
		if !strings.Contains(body, `"stop_reason":"end_turn"`) ||
			!strings.Contains(body, strings.ReplaceAll(expected, `"`, `\"`)) ||
			!strings.Contains(body, `"input_tokens":`+jsonNumber(inputTokens)) {
			t.Fatalf("unexpected structured stream: %s", body)
		}
		if strings.Contains(body, claudeStructuredOutputToolBase) || strings.Contains(body, "toolu_structured") {
			t.Fatalf("structured-output internals leaked to stream: %s", body)
		}
		return
	}

	var response ClaudeResponse
	if err := json.Unmarshal([]byte(body), &response); err != nil {
		t.Fatalf("decode Claude response: %v", err)
	}
	if response.StopReason != "end_turn" || response.Usage.InputTokens != inputTokens {
		t.Fatalf("unexpected structured response metadata: %#v", response)
	}
	if len(response.Content) != 1 || response.Content[0].Type != "text" || response.Content[0].Text != expected {
		t.Fatalf("unexpected structured response content: %#v", response.Content)
	}
}

func jsonNumber(value int) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func structuredToolNameFromRequest(t *testing.T, request KiroPayload) string {
	t.Helper()
	context := request.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if context == nil || len(context.Tools) == 0 {
		t.Fatalf("upstream request has no structured-output tool: %#v", request)
	}
	return context.Tools[len(context.Tools)-1].ToolSpecification.Name
}

func claudeStructuredOutputTestPayload() *KiroPayload {
	return ClaudeToKiro(&ClaudeRequest{
		Model: "claude-haiku-4.5",
		Messages: []ClaudeMessage{{
			Role:    "user",
			Content: "Has the stopping condition been satisfied?",
		}},
		OutputConfig: &ClaudeOutputConfig{
			Format: &ClaudeOutputFormat{
				Type: "json_schema",
				Schema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"ok": map[string]interface{}{
							"type":        "boolean",
							"description": "Whether the condition was met",
						},
						"reason": map[string]interface{}{
							"type":        "string",
							"description": "Reason, if the condition was not met",
						},
					},
					"required":             []string{"ok", "reason"},
					"additionalProperties": false,
				},
			},
		},
	}, false)
}
