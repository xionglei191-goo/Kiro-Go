package proxy

import (
	"encoding/json"
	"kiro-go/config"
	accountpool "kiro-go/pool"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestShouldAutoContinueClaudeToolCall(t *testing.T) {
	payload := claudeAutoContinueTestPayload()

	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{name: "english intent", content: "I'll inspect the current implementation first.", want: true},
		{name: "english acknowledged intent", content: "Okay, let me run the tests now.", want: true},
		{name: "english progress narration", content: "Checking the repository now.", want: true},
		{name: "chinese intent", content: "我先检查一下相关代码。", want: true},
		{name: "chinese acknowledged intent", content: "好的，我现在来重新构建并重启服务。", want: true},
		{name: "completed answer", content: "Implemented the fix and all tests pass.", want: false},
		{name: "completed chinese answer", content: "已经修复，服务运行正常。", want: false},
		{name: "intent followed by completed answer", content: "I'll inspect the files. The fix is already complete.", want: false},
		{name: "intent with dotted filename", content: "I'll inspect package.json now.", want: true},
		{name: "user instruction", content: "You can run npm test to verify the change.", want: false},
		{name: "question needs user input", content: "I'll inspect the production account now?", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldAutoContinueClaudeToolCall(payload, tt.content, nil); got != tt.want {
				t.Fatalf("shouldAutoContinueClaudeToolCall(%q) = %t, want %t", tt.content, got, tt.want)
			}
		})
	}

	if shouldAutoContinueClaudeToolCall(payload, "I'll inspect the files.", []KiroToolUse{{Name: "Bash"}}) {
		t.Fatal("must not continue after a structured tool call")
	}

	withoutTools := &KiroPayload{}
	if shouldAutoContinueClaudeToolCall(withoutTools, "I'll inspect the files.", nil) {
		t.Fatal("must not continue when the request exposes no tools")
	}

	longIntent := "I'll inspect the files. " + strings.Repeat("x", maxClaudeToolIntentRunes)
	if shouldAutoContinueClaudeToolCall(payload, longIntent, nil) {
		t.Fatal("must not classify a long final response as a stalled action")
	}
}

func TestBuildClaudeToolAutoContinuationPayload(t *testing.T) {
	payload := claudeAutoContinueTestPayload()
	payload.ToolNameMap = map[string]string{"Bash": "mcp__shell__Bash"}
	payload.ConversationState.AgentContinuationId = "original-continuation"
	payload.ConversationState.ConversationID = "conversation-1"
	payload.ConversationState.History = []KiroHistoryMessage{
		{
			UserInputMessage: &KiroUserInputMessage{
				Content: "system instructions",
				ModelID: "claude-opus-5",
				Origin:  "AI_EDITOR",
			},
		},
		{
			AssistantResponseMessage: &KiroAssistantResponseMessage{
				Content: "I will follow these instructions.",
			},
		},
		{
			UserInputMessage: &KiroUserInputMessage{
				Content: "Run the first command.",
				ModelID: "claude-opus-5",
				Origin:  "AI_EDITOR",
			},
		},
		{
			AssistantResponseMessage: &KiroAssistantResponseMessage{
				Content: "Running it.",
				ToolUses: []KiroToolUse{{
					ToolUseID: "toolu_old",
					Name:      "Bash",
					Input:     map[string]interface{}{"command": "pwd"},
				}},
			},
		},
	}
	payload.ConversationState.CurrentMessage.UserInputMessage.Content = "Command output follows."
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.ToolResults = []KiroToolResult{{
		ToolUseID: "toolu_old",
		Status:    "success",
		Content:   []KiroResultContent{{Text: "/workspace"}},
	}}

	continuation := buildClaudeToolAutoContinuationPayload(payload, "I'll inspect the files next.")
	if continuation == nil {
		t.Fatal("expected continuation payload")
	}
	if continuation.ConversationState.ConversationID != payload.ConversationState.ConversationID {
		t.Fatal("continuation must preserve the conversation ID")
	}
	if continuation.ConversationState.AgentContinuationId == payload.ConversationState.AgentContinuationId {
		t.Fatal("continuation must use a fresh agent continuation ID")
	}
	if got := continuation.ToolNameMap["Bash"]; got != "mcp__shell__Bash" {
		t.Fatalf("tool name map was not preserved: %q", got)
	}

	current := continuation.ConversationState.CurrentMessage.UserInputMessage
	if current.Content != claudeToolAutoContinuationPrompt {
		t.Fatalf("unexpected continuation instruction: %q", current.Content)
	}
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.Tools) != 1 {
		t.Fatalf("expected one tool on the continuation turn, got %#v", current.UserInputMessageContext)
	}
	if len(current.UserInputMessageContext.ToolResults) != 0 {
		t.Fatal("continuation turn must not repeat structured tool results")
	}

	var foundFirstResponse bool
	var foundNarratedResult bool
	for _, item := range continuation.ConversationState.History {
		if assistant := item.AssistantResponseMessage; assistant != nil {
			if assistant.Content == "I'll inspect the files next." {
				foundFirstResponse = true
			}
			if len(assistant.ToolUses) != 0 {
				t.Fatal("completed historical tool calls must be flattened")
			}
		}
		if user := item.UserInputMessage; user != nil {
			if strings.Contains(user.Content, "[Bash] /workspace") {
				foundNarratedResult = true
			}
			if user.UserInputMessageContext != nil {
				t.Fatal("historical turns must not retain tool context")
			}
		}
	}
	if !foundFirstResponse {
		t.Fatal("first assistant response was not added to continuation history")
	}
	if !foundNarratedResult {
		t.Fatal("previous tool result was not preserved as narrated history")
	}

	originalContext := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if originalContext == nil || len(originalContext.ToolResults) != 1 || len(originalContext.Tools) != 1 {
		t.Fatal("building a continuation mutated the original payload")
	}
}

func TestClaudeHandlersAutoContinueStalledToolIntent(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			cfgFile := t.TempDir() + "/config.json"
			if err := config.Init(cfgFile); err != nil {
				t.Fatalf("config.Init: %v", err)
			}
			if err := config.AddAccount(config.Account{
				ID:          "auto-continue-account",
				Enabled:     true,
				AccessToken: "auto-continue-token",
				ProfileArn:  "arn:aws:codewhisperer:us-east-1:123456789012:profile/test",
			}); err != nil {
				t.Fatalf("add account: %v", err)
			}
			if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
				t.Fatalf("set preferred endpoint: %v", err)
			}
			if err := config.UpdateEndpointFallback(false); err != nil {
				t.Fatalf("disable endpoint fallback: %v", err)
			}

			firstFrame := awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
				"content": "I'll inspect the files now.",
			})
			secondFrame := awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
				"toolUseId": "toolu_auto",
				"name":      "Bash",
				"input":     `{"command":"pwd"}`,
				"stop":      true,
			})

			var requests []KiroPayload
			var requestDecodeErr error
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload KiroPayload
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					requestDecodeErr = err
					http.Error(w, "invalid request", http.StatusBadRequest)
					return
				}
				requests = append(requests, payload)
				switch len(requests) {
				case 1:
					_, _ = w.Write(firstFrame)
				case 2:
					_, _ = w.Write(secondFrame)
				default:
					http.Error(w, "unexpected extra continuation", http.StatusInternalServerError)
				}
			}))
			defer server.Close()

			oldEndpoints := kiroEndpoints
			kiroEndpoints = []kiroEndpoint{{
				URL:    server.URL,
				Origin: "AI_EDITOR",
				Name:   "test",
			}}
			defer func() { kiroEndpoints = oldEndpoints }()

			oldClient := kiroHttpStore.Load()
			kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
			defer kiroHttpStore.Store(oldClient)

			pool := accountpool.GetPool()
			pool.Reload()
			handler := &Handler{
				pool:        pool,
				promptCache: newPromptCacheTracker(defaultPromptCacheTTL),
			}

			recorder := httptest.NewRecorder()
			payload := claudeAutoContinueTestPayload()
			if stream {
				handler.handleClaudeStream(recorder, payload, "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
			} else {
				handler.handleClaudeNonStream(recorder, payload, "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
			}

			if requestDecodeErr != nil {
				t.Fatalf("decode upstream request: %v", requestDecodeErr)
			}
			if len(requests) != 2 {
				t.Fatalf("expected exactly one internal continuation, got %d upstream requests", len(requests))
			}
			secondCurrent := requests[1].ConversationState.CurrentMessage.UserInputMessage
			if secondCurrent.Content != claudeToolAutoContinuationPrompt {
				t.Fatalf("unexpected continuation content: %q", secondCurrent.Content)
			}
			if secondCurrent.UserInputMessageContext == nil || len(secondCurrent.UserInputMessageContext.Tools) != 1 {
				t.Fatalf("continuation did not expose the original tool: %#v", secondCurrent.UserInputMessageContext)
			}

			if stream {
				body := recorder.Body.String()
				if !strings.Contains(body, `"stop_reason":"tool_use"`) {
					t.Fatalf("stream did not finish with tool_use: %s", body)
				}
				if !strings.Contains(body, `"id":"toolu_auto"`) {
					t.Fatalf("stream did not emit the recovered tool call: %s", body)
				}
				return
			}

			var response ClaudeResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode Claude response: %v", err)
			}
			if response.StopReason != "tool_use" {
				t.Fatalf("expected tool_use stop reason, got %q", response.StopReason)
			}
			if len(response.Content) < 2 || response.Content[len(response.Content)-1].Name != "Bash" {
				t.Fatalf("expected recovered Bash tool call, got %#v", response.Content)
			}
		})
	}
}

func claudeAutoContinueTestPayload() *KiroPayload {
	payload := &KiroPayload{}
	payload.ConversationState.AgentContinuationId = "continuation-1"
	payload.ConversationState.ConversationID = "conversation-1"
	payload.ConversationState.ChatTriggerType = "MANUAL"
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "Inspect the repository and fix the issue.",
		ModelID: "claude-opus-5",
		Origin:  "AI_EDITOR",
		UserInputMessageContext: &UserInputMessageContext{
			Tools: []KiroToolWrapper{{}},
		},
	}
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools[0].ToolSpecification.Name = "Bash"
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools[0].ToolSpecification.Description = "Run a shell command."
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools[0].ToolSpecification.InputSchema.JSON = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{"type": "string"},
		},
	}
	return payload
}
