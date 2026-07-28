package proxy

import (
	"encoding/json"
	"kiro-go/config"
	accountpool "kiro-go/pool"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestShouldRunClaudeToolControllerUsesProtocolStateNotResponseText(t *testing.T) {
	payload := claudeToolControllerTestPayload()

	for _, content := range []string{
		"I'll inspect the current implementation first.",
		"Emiting the checks now.",
		"Deploy is in its local gate phase. Monitoring until it finishes.",
		"The deployment completed successfully.",
		"需要用户选择部署区域吗？",
		"",
	} {
		if !shouldRunClaudeToolController(payload, nil) {
			t.Fatalf("controller eligibility must not depend on response wording: %q", content)
		}
	}

	if shouldRunClaudeToolController(payload, []KiroToolUse{{Name: "Bash"}}) {
		t.Fatal("controller must not run after a real structured tool call")
	}

	payload.ClaudeCodeAgent = false
	if shouldRunClaudeToolController(payload, nil) {
		t.Fatal("controller must not run for a generic Anthropic client")
	}

	payload.ClaudeCodeAgent = true
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext = nil
	if shouldRunClaudeToolController(payload, nil) {
		t.Fatal("controller must not run when no real tools are available")
	}
}

func TestClaudeAssistantRequestsUserInput(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{
			name:    "production confirmation regression",
			content: "改动还未提交。我在等你确认是否提交并部署。",
			want:    true,
		},
		{
			name:    "chinese direct question",
			content: "当前提交仍在本地。要我撤销还是保留？",
			want:    true,
		},
		{
			name:    "english approval",
			content: "The migration is ready. I am waiting for your approval before applying it.",
			want:    true,
		},
		{
			name:    "english direct question",
			content: "Would you like me to deploy this build now?",
			want:    true,
		},
		{
			name: "question inside code fence",
			content: "The command printed:\n```\nProceed? [y/N]\n```\n" +
				"I will inspect the non-interactive flags next.",
			want: false,
		},
		{
			name:    "autonomous whether statement",
			content: "I need to know whether the service is healthy, so I will run the health check.",
			want:    false,
		},
		{
			name:    "normal continuation",
			content: "The build passed. I will run the remaining integration tests.",
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := claudeAssistantRequestsUserInput(tt.content); got != tt.want {
				t.Fatalf("claudeAssistantRequestsUserInput() = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestBuildClaudeToolControllerPayload(t *testing.T) {
	payload := claudeToolControllerTestPayload()
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

	controller := buildClaudeToolControllerPayload(payload, "Deploy is still running.")
	if controller == nil {
		t.Fatal("expected controller payload")
	}
	if controller.ConversationState.ConversationID != payload.ConversationState.ConversationID {
		t.Fatal("controller must preserve the conversation ID")
	}
	if controller.ConversationState.AgentContinuationId == payload.ConversationState.AgentContinuationId {
		t.Fatal("controller must use a fresh agent continuation ID")
	}
	if got := controller.ToolNameMap["Bash"]; got != "mcp__shell__Bash" {
		t.Fatalf("tool name map was not preserved: %q", got)
	}
	if !controller.ClaudeCodeAgent {
		t.Fatal("controller marker was not preserved")
	}
	if controller.ControllerFinishToolName == "" || controller.ControllerWaitToolName == "" {
		t.Fatal("controller decision tool names were not assigned")
	}

	current := controller.ConversationState.CurrentMessage.UserInputMessage
	if !strings.Contains(current.Content, controller.ControllerFinishToolName) ||
		!strings.Contains(current.Content, controller.ControllerWaitToolName) ||
		!strings.Contains(current.Content, "without emitting text") ||
		!strings.Contains(current.Content, "response is invalid unless") ||
		!strings.Contains(current.Content, "Safety takes priority") {
		t.Fatalf("unexpected controller instruction: %q", current.Content)
	}
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.Tools) != 3 {
		t.Fatalf("expected one real and two internal tools, got %#v", current.UserInputMessageContext)
	}
	if controller.InferenceConfig == nil || controller.InferenceConfig.MaxTokens != claudeControllerMaxTokens {
		t.Fatalf("controller max tokens were not capped: %#v", controller.InferenceConfig)
	}
	if len(current.UserInputMessageContext.ToolResults) != 0 {
		t.Fatal("controller turn must not repeat structured tool results")
	}

	toolNames := make(map[string]bool)
	for _, tool := range current.UserInputMessageContext.Tools {
		toolNames[tool.ToolSpecification.Name] = true
	}
	for _, name := range []string{"Bash", controller.ControllerFinishToolName, controller.ControllerWaitToolName} {
		if !toolNames[name] {
			t.Fatalf("controller tool %q is missing from %#v", name, toolNames)
		}
	}

	var foundFirstResponse bool
	var foundNarratedResult bool
	for _, item := range controller.ConversationState.History {
		if assistant := item.AssistantResponseMessage; assistant != nil {
			if assistant.Content == "Deploy is still running." {
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
		t.Fatal("first assistant response was not added to controller history")
	}
	if !foundNarratedResult {
		t.Fatal("previous tool result was not preserved as narrated history")
	}

	originalContext := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if originalContext == nil || len(originalContext.ToolResults) != 1 || len(originalContext.Tools) != 1 {
		t.Fatal("building a controller request mutated the original payload")
	}
	if payload.ControllerFinishToolName != "" || payload.ControllerWaitToolName != "" {
		t.Fatal("building a controller request mutated original controller metadata")
	}
}

func TestClaudeControllerModelOverride(t *testing.T) {
	payload := claudeToolControllerTestPayload()

	t.Run("defaults to primary", func(t *testing.T) {
		t.Setenv(claudeControllerModelEnv, "")
		controller := buildClaudeToolControllerPayload(payload, "Finished.")
		if got := currentMessageModelID(controller); got != "claude-opus-5" {
			t.Fatalf("controller model = %q, want primary model", got)
		}
	})

	t.Run("supports normalized Claude model", func(t *testing.T) {
		t.Setenv(claudeControllerModelEnv, "claude-haiku-4-5")
		controller := buildClaudeToolControllerPayload(payload, "Finished.")
		if got := currentMessageModelID(controller); got != "claude-haiku-4.5" {
			t.Fatalf("controller model = %q, want claude-haiku-4.5", got)
		}
	})

	t.Run("invalid model falls back to primary", func(t *testing.T) {
		t.Setenv(claudeControllerModelEnv, "not-a-claude-model")
		controller := buildClaudeToolControllerPayload(payload, "Finished.")
		if got := currentMessageModelID(controller); got != "claude-opus-5" {
			t.Fatalf("controller model = %q, want primary fallback", got)
		}
	})
}

func TestClaudeControllerPreservesSmallerMaxTokens(t *testing.T) {
	payload := claudeToolControllerTestPayload()
	payload.InferenceConfig = &InferenceConfig{MaxTokens: 256}

	controller := buildClaudeToolControllerPayload(payload, "Finished.")
	if controller.InferenceConfig == nil || controller.InferenceConfig.MaxTokens != 256 {
		t.Fatalf("controller max tokens = %#v, want 256", controller.InferenceConfig)
	}
	if payload.InferenceConfig.MaxTokens != 256 {
		t.Fatalf("building controller mutated primary inference config: %#v", payload.InferenceConfig)
	}
}

func TestClaudeControllerCompactsOversizedHistoryToAuxiliaryLimit(t *testing.T) {
	t.Setenv(claudeControllerModelEnv, "claude-haiku-4.5")
	payload := claudeToolControllerTestPayload()
	payload.ControllerOriginalTask = "ORIGINAL TASK: repair the deployment and verify it."
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
	}
	largeTurn := strings.Repeat("repository output and implementation detail ", 5000)
	for i := 0; i < 12; i++ {
		payload.ConversationState.History = append(
			payload.ConversationState.History,
			KiroHistoryMessage{UserInputMessage: &KiroUserInputMessage{
				Content: largeTurn,
				ModelID: "claude-opus-5",
				Origin:  "AI_EDITOR",
			}},
			KiroHistoryMessage{AssistantResponseMessage: &KiroAssistantResponseMessage{
				Content: largeTurn,
			}},
		)
	}
	primarySize := payloadByteSize(payload)

	controller := buildClaudeToolControllerPayload(payload, "Run the remaining verification.")
	if controller == nil {
		t.Fatal("expected controller payload")
	}
	if got := payloadByteSize(controller); got > claudeControllerMaxPayloadBytes {
		t.Fatalf("controller payload size = %d, limit = %d", got, claudeControllerMaxPayloadBytes)
	}
	tokenBudget := claudeControllerInputTokenBudget(payload, false)
	if got := estimateKiroPayloadTokens(controller); got > tokenBudget {
		t.Fatalf("controller token estimate = %d, limit = %d", got, tokenBudget)
	}
	if primarySize <= claudeControllerMaxPayloadBytes {
		t.Fatalf("test payload must exceed controller limit, got %d", primarySize)
	}
	if got := payloadByteSize(payload); got != primarySize {
		t.Fatalf("building controller mutated primary payload size: got %d, want %d", got, primarySize)
	}

	foundPlaceholder := false
	for _, item := range controller.ConversationState.History {
		if user := item.UserInputMessage; user != nil &&
			strings.Contains(user.Content, "controller-only history was omitted") {
			foundPlaceholder = true
		}
	}
	if !foundPlaceholder {
		t.Fatal("expected controller history truncation marker")
	}
	for i := 1; i < len(controller.ConversationState.History); i++ {
		previousUser := controller.ConversationState.History[i-1].UserInputMessage != nil
		currentUser := controller.ConversationState.History[i].UserInputMessage != nil
		if previousUser == currentUser {
			t.Fatalf("controller history roles do not alternate at index %d", i)
		}
	}
	currentPrompt := controller.ConversationState.CurrentMessage.UserInputMessage.Content
	if !strings.Contains(currentPrompt, "ORIGINAL TASK: repair the deployment") ||
		!strings.Contains(currentPrompt, "Run the remaining verification.") {
		t.Fatalf("controller lost deterministic context anchors: %q", currentPrompt)
	}
}

func TestClaudeControllerCompactsByTokenDensityBeforeByteLimit(t *testing.T) {
	t.Setenv(claudeControllerModelEnv, "claude-haiku-4.5")
	payload := claudeToolControllerTestPayload()
	payload.ControllerOriginalTask = "Keep running checks until the service is healthy."
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
	}
	denseTurn := strings.Repeat("{}[],:;!@#$%^&*", 2500)
	for i := 0; i < 4; i++ {
		payload.ConversationState.History = append(
			payload.ConversationState.History,
			KiroHistoryMessage{UserInputMessage: &KiroUserInputMessage{
				Content: denseTurn,
				ModelID: "claude-opus-5",
				Origin:  "AI_EDITOR",
			}},
			KiroHistoryMessage{AssistantResponseMessage: &KiroAssistantResponseMessage{
				Content: denseTurn,
			}},
		)
	}
	if got := payloadByteSize(payload); got >= claudeControllerMaxPayloadBytes {
		t.Fatalf("test payload must be below byte limit, got %d", got)
	}

	controller := buildClaudeToolControllerPayload(payload, "Continue with the next check.")
	if controller == nil {
		t.Fatal("expected controller payload")
	}
	tokenBudget := claudeControllerInputTokenBudget(payload, false)
	if got := estimateKiroPayloadTokens(controller); got > tokenBudget {
		t.Fatalf("controller token estimate = %d, limit = %d", got, tokenBudget)
	}
	if got := payloadByteSize(controller); got > claudeControllerMaxPayloadBytes {
		t.Fatalf("controller payload size = %d, limit = %d", got, claudeControllerMaxPayloadBytes)
	}
	currentPrompt := controller.ConversationState.CurrentMessage.UserInputMessage.Content
	if !strings.Contains(currentPrompt, "Keep running checks") ||
		!strings.Contains(currentPrompt, "Continue with the next check") {
		t.Fatalf("controller token compaction lost anchors: %q", currentPrompt)
	}
}

func TestClaudeControllerRetriesOnceWithFallbackBudgetOnContentLength(t *testing.T) {
	t.Setenv(claudeControllerModelEnv, "claude-haiku-4.5")
	cfgFile := t.TempDir() + "/config.json"
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set preferred endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(true); err != nil {
		t.Fatalf("enable endpoint fallback: %v", err)
	}

	var mu sync.Mutex
	var requests []KiroPayload
	primaryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload KiroPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests = append(requests, payload)
		call := len(requests)
		mu.Unlock()
		if call == 1 {
			http.Error(w, `{"message":"Input is too long.","reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}`, http.StatusBadRequest)
			return
		}
		_, _ = w.Write(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "toolu_retry",
			"name":      "Bash",
			"input":     `{"command":"check-health"}`,
			"stop":      true,
		}))
	}))
	defer primaryServer.Close()

	var unexpectedFallbackCalls int
	unexpectedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		unexpectedFallbackCalls++
		mu.Unlock()
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer unexpectedServer.Close()

	oldEndpoints := kiroEndpoints
	kiroEndpoints = []kiroEndpoint{
		{URL: primaryServer.URL, Origin: "AI_EDITOR", Name: "first"},
		{URL: unexpectedServer.URL, Origin: "AI_EDITOR", Name: "second"},
		{URL: unexpectedServer.URL, Origin: "AI_EDITOR", Name: "third"},
	}
	defer func() { kiroEndpoints = oldEndpoints }()

	oldClient := kiroHttpStore.Load()
	kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
	defer kiroHttpStore.Store(oldClient)

	source := claudeToolControllerTestPayload()
	source.ControllerOriginalTask = "Verify the deployment."
	source.ConversationState.History = append(source.ConversationState.History,
		KiroHistoryMessage{UserInputMessage: &KiroUserInputMessage{
			Content: strings.Repeat("long context ", 50_000),
			ModelID: "claude-opus-5",
			Origin:  "AI_EDITOR",
		}},
	)
	initial := buildClaudeToolControllerPayload(source, "More checks remain.")
	account := &config.Account{
		ID:          "controller-fallback",
		AccessToken: "token",
		ProfileArn:  "arn:aws:codewhisperer:us-east-1:123456789012:profile/test",
	}
	var attempts []claudeControllerRetryReason
	usedPayload, toolUses, retryReason, err := callClaudeToolControllerWithFallback(
		account,
		source,
		"More checks remain.",
		initial,
		&kiroCallMetrics{},
		func(_ *KiroPayload, reason claudeControllerRetryReason) {
			attempts = append(attempts, reason)
		},
	)
	if err != nil {
		t.Fatalf("controller fallback failed: %v", err)
	}
	if retryReason != claudeControllerRetryContentLength ||
		len(attempts) != 2 ||
		attempts[0] != claudeControllerNoRetry ||
		attempts[1] != claudeControllerRetryContentLength {
		t.Fatalf("unexpected fallback attempts: reason=%s attempts=%v", retryReason, attempts)
	}
	mu.Lock()
	unexpectedCalls := unexpectedFallbackCalls
	mu.Unlock()
	if unexpectedCalls != 0 {
		t.Fatalf("content-length error should not fan out across endpoints, got %d calls", unexpectedCalls)
	}
	if len(requests) != 2 {
		t.Fatalf("expected initial call plus one compact retry, got %d", len(requests))
	}
	fallbackTokenBudget := claudeControllerInputTokenBudget(source, true)
	if got := estimateKiroPayloadTokens(usedPayload); got > fallbackTokenBudget {
		t.Fatalf("fallback token estimate = %d, limit = %d", got, fallbackTokenBudget)
	}
	if got := payloadByteSize(usedPayload); got > claudeControllerFallbackPayloadBytes {
		t.Fatalf("fallback payload size = %d, limit = %d", got, claudeControllerFallbackPayloadBytes)
	}
	if len(toolUses) != 1 || toolUses[0].Name != "Bash" {
		t.Fatalf("fallback tool result was not returned: %#v", toolUses)
	}
	if usedPayload.InferenceConfig == nil ||
		usedPayload.InferenceConfig.MaxTokens != claudeControllerFallbackMaxTokens {
		t.Fatalf("fallback max tokens were not tightened: %#v", usedPayload.InferenceConfig)
	}
	if !strings.Contains(
		usedPayload.ConversationState.CurrentMessage.UserInputMessage.Content,
		"final controller decision attempt",
	) {
		t.Fatal("fallback controller prompt was not tightened")
	}
	if !strings.Contains(
		usedPayload.ConversationState.CurrentMessage.UserInputMessage.Content,
		"Safety remains higher priority",
	) {
		t.Fatal("fallback controller prompt lost its safety priority")
	}
}

func TestClaudeControllerTokenBudgetTracksControllerModelWindow(t *testing.T) {
	payload := claudeToolControllerTestPayload()

	t.Run("haiku 200k window", func(t *testing.T) {
		t.Setenv(claudeControllerModelEnv, "claude-haiku-4.5")
		if got := claudeControllerInputTokenBudget(payload, false); got != 150_000 {
			t.Fatalf("normal Haiku budget = %d, want 150000", got)
		}
		if got := claudeControllerInputTokenBudget(payload, true); got != 90_000 {
			t.Fatalf("fallback Haiku budget = %d, want 90000", got)
		}
	})

	t.Run("opus 1m window is capped", func(t *testing.T) {
		t.Setenv(claudeControllerModelEnv, "same")
		payload.ConversationState.CurrentMessage.UserInputMessage.ModelID = "claude-opus-4.8"
		if got := claudeControllerInputTokenBudget(payload, false); got != 300_000 {
			t.Fatalf("normal Opus budget = %d, want 300000", got)
		}
		if got := claudeControllerInputTokenBudget(payload, true); got != 150_000 {
			t.Fatalf("fallback Opus budget = %d, want 150000", got)
		}
	})
}

func TestClaudeControllerToolNamesAvoidClientCollisions(t *testing.T) {
	payload := claudeToolControllerTestPayload()
	context := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	context.Tools = append(
		context.Tools,
		newClaudeControllerTool(claudeControllerFinishToolBase, "client-owned collision"),
		newClaudeControllerTool(claudeControllerWaitToolBase, "client-owned collision"),
	)

	controller := buildClaudeToolControllerPayload(payload, "status")
	if controller == nil {
		t.Fatal("expected controller payload")
	}
	if controller.ControllerFinishToolName == claudeControllerFinishToolBase {
		t.Fatal("finish tool name collided with a client tool")
	}
	if controller.ControllerWaitToolName == claudeControllerWaitToolBase {
		t.Fatal("wait tool name collided with a client tool")
	}
}

func TestClaudeControllerHonorsRequiredToolChoice(t *testing.T) {
	payload := claudeToolControllerTestPayload()
	payload.ClaudeToolChoiceRequired = true

	controller := buildClaudeToolControllerPayload(payload, "I will run the command.")
	if controller == nil {
		t.Fatal("expected controller payload")
	}
	if !controller.ClaudeToolChoiceRequired {
		t.Fatal("required tool choice was not preserved")
	}
	if controller.ControllerFinishToolName != "" || controller.ControllerWaitToolName != "" {
		t.Fatal("required tool choice must not expose internal end-turn decisions")
	}
	current := controller.ConversationState.CurrentMessage.UserInputMessage
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.Tools) != 1 {
		t.Fatalf("required tool choice must expose only the selected real tools: %#v", current.UserInputMessageContext)
	}
	if !strings.Contains(current.Content, "requires a real tool call") {
		t.Fatalf("required tool choice prompt is missing enforcement: %q", current.Content)
	}
}

func TestSplitClaudeControllerToolUses(t *testing.T) {
	payload := &KiroPayload{
		ControllerFinishToolName: "finish",
		ControllerWaitToolName:   "wait",
	}
	real := KiroToolUse{ToolUseID: "toolu_real", Name: "Bash"}

	tests := []struct {
		name       string
		toolUses   []KiroToolUse
		wantReal   int
		wantResult claudeControllerOutcome
	}{
		{name: "continue", toolUses: []KiroToolUse{real}, wantReal: 1, wantResult: claudeControllerContinue},
		{name: "finish", toolUses: []KiroToolUse{{Name: "finish"}}, wantResult: claudeControllerFinish},
		{name: "wait", toolUses: []KiroToolUse{{Name: "wait"}}, wantResult: claudeControllerWaitForUser},
		{name: "real tool wins conflict", toolUses: []KiroToolUse{{Name: "finish"}, real}, wantReal: 1, wantResult: claudeControllerContinue},
		{name: "undecided", wantResult: claudeControllerUndecided},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotReal, gotResult := splitClaudeControllerToolUses(payload, tt.toolUses)
			if len(gotReal) != tt.wantReal || gotResult != tt.wantResult {
				t.Fatalf("split = (%d, %s), want (%d, %s)", len(gotReal), gotResult, tt.wantReal, tt.wantResult)
			}
		})
	}
}

func TestClaudeHandlersControllerContinuesWithRealTool(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, _ KiroPayload) []byte {
				switch call {
				case 1:
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content":     "Deploy is in its local gate phase. Monitoring until it finishes.",
						"inputTokens": 100,
					})
				case 2:
					return append(
						awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
							"content": "This controller text must stay hidden.",
						}),
						awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
							"toolUseId":   "toolu_controller",
							"name":        "Bash",
							"input":       `{"command":"check-deploy"}`,
							"stop":        true,
							"inputTokens": 120,
						})...,
					)
				default:
					return nil
				}
			})

			if len(requests) != 2 {
				t.Fatalf("expected one controller request, got %d upstream requests", len(requests))
			}
			assertControllerRequestShape(t, requests[1])

			if stream {
				body := recorder.Body.String()
				if !strings.Contains(body, `"stop_reason":"tool_use"`) ||
					!strings.Contains(body, `"id":"toolu_controller"`) ||
					!strings.Contains(body, `"input_tokens":220`) {
					t.Fatalf("stream did not emit the controller's real tool call: %s", body)
				}
				if strings.Contains(body, "controller text must stay hidden") ||
					strings.Contains(body, claudeControllerFinishToolBase) ||
					strings.Contains(body, claudeControllerWaitToolBase) {
					t.Fatalf("stream leaked internal controller output: %s", body)
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
			if response.Usage.InputTokens != 220 {
				t.Fatalf("expected aggregated controller input usage, got %#v", response.Usage)
			}
			if len(response.Content) < 2 || response.Content[len(response.Content)-1].Name != "Bash" {
				t.Fatalf("expected recovered Bash tool call, got %#v", response.Content)
			}
			for _, block := range response.Content {
				if strings.Contains(block.Text, "controller text must stay hidden") ||
					block.Name == claudeControllerFinishToolBase ||
					block.Name == claudeControllerWaitToolBase {
					t.Fatalf("response leaked internal controller output: %#v", response.Content)
				}
			}
		})
	}
}

func TestClaudeHandlersPreserveExplicitUserConfirmation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			const confirmation = "改动还未提交。我在等你确认是否提交并部署。"
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, _ KiroPayload) []byte {
				if call == 1 {
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content":     confirmation,
						"inputTokens": 100,
					})
				}
				return awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
					"toolUseId": "toolu_unauthorized",
					"name":      "Bash",
					"input":     `{"command":"git push origin main"}`,
					"stop":      true,
				})
			})

			if len(requests) != 1 {
				t.Fatalf("explicit confirmation request triggered %d upstream calls, want 1", len(requests))
			}
			body := recorder.Body.String()
			if !strings.Contains(body, confirmation) ||
				!strings.Contains(body, `"stop_reason":"end_turn"`) {
				t.Fatalf("confirmation request was not preserved as end_turn: %s", body)
			}
			if strings.Contains(body, "toolu_unauthorized") ||
				strings.Contains(body, "git push origin main") {
				t.Fatalf("confirmation request leaked an unauthorized tool call: %s", body)
			}
		})
	}
}

func TestClaudeHandlersControllerRetriesUndecidedDecision(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, _ KiroPayload) []byte {
				switch call {
				case 1:
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content":     "Original response.",
						"inputTokens": 100,
					})
				case 2:
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content":     "I still decline to choose a tool.",
						"inputTokens": 120,
					})
				case 3:
					return awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
						"toolUseId":   "toolu_retry_decision",
						"name":        "Bash",
						"input":       `{"command":"continue-work"}`,
						"stop":        true,
						"inputTokens": 80,
					})
				default:
					return nil
				}
			})

			if len(requests) != 3 {
				t.Fatalf("expected one strict decision retry, got %d requests", len(requests))
			}
			retry := requests[2]
			if retry.InferenceConfig == nil ||
				retry.InferenceConfig.MaxTokens != claudeControllerFallbackMaxTokens ||
				!strings.Contains(
					retry.ConversationState.CurrentMessage.UserInputMessage.Content,
					"final controller decision attempt",
				) {
				t.Fatalf("strict controller retry was not applied: %#v", retry)
			}

			body := recorder.Body.String()
			if !strings.Contains(body, `"stop_reason":"tool_use"`) ||
				!strings.Contains(body, "toolu_retry_decision") {
				t.Fatalf("strict retry did not recover the tool call: %s", body)
			}
			if stream {
				if !strings.Contains(body, `"input_tokens":300`) {
					t.Fatalf("stream did not aggregate both controller attempts: %s", body)
				}
			} else {
				var response ClaudeResponse
				if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
					t.Fatalf("decode response: %v", err)
				}
				if response.Usage.InputTokens != 300 {
					t.Fatalf("controller retry usage = %#v, want 300 input tokens", response.Usage)
				}
			}
			if strings.Contains(body, "I still decline to choose a tool.") {
				t.Fatalf("controller retry text leaked to the client: %s", body)
			}
		})
	}
}

func TestClaudeHandlerStopsAfterNonRetryableControllerError(t *testing.T) {
	recorder, requests := runClaudeControllerHandlerTest(t, true, func(call int, _ KiroPayload) []byte {
		if call == 1 {
			return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
				"content": "Original response.",
			})
		}
		if call == 2 {
			return []byte{0x01}
		}
		return nil
	})

	if len(requests) != 2 {
		t.Fatalf("non-retryable controller error made extra requests: %d", len(requests))
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"stop_reason":"end_turn"`) ||
		!strings.Contains(body, "Original response.") {
		t.Fatalf("controller error did not preserve original response: %s", body)
	}
	if strings.Contains(body, "event: error") {
		t.Fatalf("controller internal error leaked to the client: %s", body)
	}
}

func TestClaudeHandlersControllerInterceptsFinish(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-stream"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			recorder, requests := runClaudeControllerHandlerTest(t, stream, func(call int, _ KiroPayload) []byte {
				switch call {
				case 1:
					return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
						"content": "Deployment completed successfully.",
					})
				case 2:
					return awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
						"toolUseId": "toolu_finish",
						"name":      claudeControllerFinishToolBase,
						"input":     `{"reason":"deployment and checks completed"}`,
						"stop":      true,
					})
				default:
					return nil
				}
			})

			if len(requests) != 2 {
				t.Fatalf("expected one controller request, got %d upstream requests", len(requests))
			}
			if stream {
				body := recorder.Body.String()
				if !strings.Contains(body, `"stop_reason":"end_turn"`) ||
					!strings.Contains(body, "Deployment completed successfully.") {
					t.Fatalf("stream did not preserve the completed answer: %s", body)
				}
				if strings.Contains(body, claudeControllerFinishToolBase) || strings.Contains(body, "toolu_finish") {
					t.Fatalf("stream leaked the internal finish tool: %s", body)
				}
				return
			}

			var response ClaudeResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("decode Claude response: %v", err)
			}
			if response.StopReason != "end_turn" {
				t.Fatalf("expected end_turn after finish decision, got %q", response.StopReason)
			}
			if len(response.Content) != 1 || response.Content[0].Text != "Deployment completed successfully." {
				t.Fatalf("finish decision changed the client response: %#v", response.Content)
			}
		})
	}
}

func TestClaudeHandlerSkipsControllerForGenericClient(t *testing.T) {
	recorder, requests := runClaudeControllerHandlerTest(t, false, func(call int, _ KiroPayload) []byte {
		if call == 1 {
			return awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
				"content": "A normal final answer.",
			})
		}
		return nil
	}, func(payload *KiroPayload) {
		payload.ClaudeCodeAgent = false
	})

	if len(requests) != 1 {
		t.Fatalf("generic client unexpectedly triggered controller: %d requests", len(requests))
	}
	var response ClaudeResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode Claude response: %v", err)
	}
	if response.StopReason != "end_turn" {
		t.Fatalf("expected generic end_turn, got %q", response.StopReason)
	}
}

type claudeControllerResponder func(call int, payload KiroPayload) []byte

func runClaudeControllerHandlerTest(
	t *testing.T,
	stream bool,
	responder claudeControllerResponder,
	mutators ...func(*KiroPayload),
) (*httptest.ResponseRecorder, []KiroPayload) {
	t.Helper()

	cfgFile := t.TempDir() + "/config.json"
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	accountID := "controller-account-" + strings.NewReplacer("/", "-", " ", "-").Replace(t.Name())
	if err := config.AddAccount(config.Account{
		ID:          accountID,
		Enabled:     true,
		AccessToken: "controller-token",
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

	var mu sync.Mutex
	var requests []KiroPayload
	var requestDecodeErr error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload KiroPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			mu.Lock()
			requestDecodeErr = err
			mu.Unlock()
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests = append(requests, payload)
		call := len(requests)
		mu.Unlock()
		frame := responder(call, payload)
		if len(frame) == 0 {
			http.Error(w, "unexpected extra request", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write(frame)
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
	payload := claudeToolControllerTestPayload()
	for _, mutate := range mutators {
		mutate(payload)
	}
	if stream {
		handler.handleClaudeStream(recorder, payload, "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
	} else {
		handler.handleClaudeNonStream(recorder, payload, "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
	}

	mu.Lock()
	defer mu.Unlock()
	if requestDecodeErr != nil {
		t.Fatalf("decode upstream request: %v", requestDecodeErr)
	}
	return recorder, append([]KiroPayload(nil), requests...)
}

func assertControllerRequestShape(t *testing.T, request KiroPayload) {
	t.Helper()
	current := request.ConversationState.CurrentMessage.UserInputMessage
	if !strings.Contains(current.Content, claudeControllerFinishToolBase) ||
		!strings.Contains(current.Content, claudeControllerWaitToolBase) {
		t.Fatalf("controller prompt does not name its decision tools: %q", current.Content)
	}
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.Tools) != 3 {
		t.Fatalf("expected one real and two controller tools: %#v", current.UserInputMessageContext)
	}
}

func claudeToolControllerTestPayload() *KiroPayload {
	payload := &KiroPayload{ClaudeCodeAgent: true}
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
