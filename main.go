// openClaude: Anthropic Messages API → OpenAI Chat Completions proxy.
// Use any OpenAI-compatible model with Claude Code.
//
// Build:  go build -o openClaude .
// Run:    openClaude --model qwen3-max    — start proxy + Claude Code
//         openClaude serve                — start proxy only
//         openClaude list                 — show available models
//
// Config (config.json):
//   providers: map of provider name → { url, apiKeyEnv }
//   models:    map of alias → "provider/upstream-model-name"
//
// Models not found in config are passed through to the Anthropic API directly,
// so native Claude models always work without any configuration.

package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ─── Configuration ──────────────────────────────────────────

type Config struct {
	Port      int                 `json:"port"`
	Providers map[string]Provider `json:"providers"`
	Models    map[string]string   `json:"models"` // alias → "provider/model-id"
}

type Provider struct {
	URL       string `json:"url"`
	APIKeyEnv string `json:"apiKeyEnv,omitempty"`
	APIKeyCmd string `json:"apiKeyCmd,omitempty"`
	apiKey    string
}

var (
	cfg          Config
	anthropicAPI = "https://api.anthropic.com"
)

func loadConfig() {
	cfg.Port = 8082

	configPath := os.Getenv("OCC_CONFIG")
	if configPath == "" {
		// Search order: next to binary, current dir, ~/.config/openClaude/
		candidates := []string{"config.json"}
		if exe, err := os.Executable(); err == nil {
			candidates = append([]string{filepath.Join(filepath.Dir(exe), "config.json")}, candidates...)
		}
		if home, err := os.UserHomeDir(); err == nil {
			candidates = append(candidates, filepath.Join(home, ".config", "openClaude", "config.json"))
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				configPath = c
				break
			}
		}
		if configPath == "" {
			configPath = candidates[len(candidates)-1] // for error message
		}
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("Cannot read config file: %v\n\nCreate one at ~/.config/openClaude/config.json\nSee config.example.json for the format.", err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		log.Fatalf("Invalid config %s: %v", configPath, err)
	}

	if cfg.Port == 0 {
		cfg.Port = 8082
	}
	// Override port from env if set
	if p := os.Getenv("PORT"); p != "" {
		fmt.Sscanf(p, "%d", &cfg.Port)
	}

	if len(cfg.Providers) == 0 {
		log.Fatal("No providers configured. Add at least one provider to config.json")
	}

	// Resolve API keys: apiKeyCmd takes priority, then apiKeyEnv
	for name, p := range cfg.Providers {
		if p.APIKeyCmd != "" {
			out, err := exec.Command("sh", "-c", p.APIKeyCmd).Output()
			if err != nil {
				log.Fatalf("Provider %q: apiKeyCmd failed: %v", name, err)
			}
			p.apiKey = strings.TrimSpace(string(out))
		} else if p.APIKeyEnv != "" {
			p.apiKey = os.Getenv(p.APIKeyEnv)
		}
		if p.apiKey == "" {
			log.Fatalf("Provider %q: no API key found (set apiKeyCmd or apiKeyEnv in config)", name)
		}
		if p.URL == "" {
			log.Fatalf("Provider %q missing url field", name)
		}
		cfg.Providers[name] = p
	}

	// Validate model routes
	for alias, route := range cfg.Models {
		provName, _, ok := splitRoute(route)
		if !ok {
			log.Fatalf("Invalid model route %q for %q — expected \"provider/model-name\"", route, alias)
		}
		if _, exists := cfg.Providers[provName]; !exists {
			log.Fatalf("Model %q references unknown provider %q", alias, provName)
		}
	}
}

// splitRoute splits "provider/model-id" into provider name and model ID.
// Uses first "/" as delimiter so model IDs like "Qwen/Qwen3-Max" work.
func splitRoute(route string) (provider, model string, ok bool) {
	i := strings.Index(route, "/")
	if i < 1 || i >= len(route)-1 {
		return "", "", false
	}
	return route[:i], route[i+1:], true
}

// resolveModel finds the provider and upstream model name for a given model alias.
// Returns nil provider if the model should be passed through to Anthropic.
func resolveModel(model string) (*Provider, string) {
	// Exact match first
	if route, ok := cfg.Models[model]; ok {
		provName, upModel, _ := splitRoute(route)
		p := cfg.Providers[provName]
		return &p, upModel
	}

	// Case-insensitive contains match, longest alias first
	lower := strings.ToLower(model)
	aliases := make([]string, 0, len(cfg.Models))
	for k := range cfg.Models {
		aliases = append(aliases, k)
	}
	sort.Slice(aliases, func(i, j int) bool { return len(aliases[i]) > len(aliases[j]) })

	for _, alias := range aliases {
		if strings.Contains(lower, strings.ToLower(alias)) {
			provName, upModel, _ := splitRoute(cfg.Models[alias])
			p := cfg.Providers[provName]
			return &p, upModel
		}
	}

	return nil, ""
}

func generateID(prefix string) string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("%s_%x", prefix, b)
}

func ptr(s string) *string { return &s }

// ─── Anthropic types ────────────────────────────────────────

type MessagesRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	Messages      []MessageParam  `json:"messages"`
	System        any             `json:"system,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Tools         []AnthropicTool `json:"tools,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	Thinking      *ThinkingConfig `json:"thinking,omitempty"`
}

type MessageParam struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

func (m *MessageParam) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	if m.Role == "" {
		m.Role = "user"
	}

	if raw.Content == nil || string(raw.Content) == "null" {
		m.Content = nil
		return nil
	}

	// Content can be a plain string or array of blocks
	var s string
	if err := json.Unmarshal(raw.Content, &s); err == nil {
		m.Content = []ContentBlock{{Type: "text", Text: ptr(s)}}
		return nil
	}

	return json.Unmarshal(raw.Content, &m.Content)
}

type ContentBlock struct {
	Type      string          `json:"type"`
	Text      *string         `json:"text,omitempty"`
	Source    *ImageSource    `json:"source,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   any             `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Thinking  *string         `json:"thinking,omitempty"`
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type ToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type ThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type AnthropicResponse struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Model      string         `json:"model"`
	Content    []ContentBlock `json:"content"`
	StopReason string         `json:"stop_reason,omitempty"`
	Usage      AnthropicUsage `json:"usage"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type AnthropicError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// ─── OpenAI types ───────────────────────────────────────────

type OpenAIRequest struct {
	Model         string             `json:"model"`
	Messages      []OpenAIMessage    `json:"messages"`
	MaxTokens     int                `json:"max_tokens,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	Stream        bool               `json:"stream"`
	StreamOptions *OpenAIStreamOpts  `json:"stream_options,omitempty"`
	Stop          []string           `json:"stop,omitempty"`
	Tools         []OpenAITool       `json:"tools,omitempty"`
	ToolChoice    any                `json:"tool_choice,omitempty"`
}

type OpenAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type OpenAIMessage struct {
	Role             string          `json:"role"`
	Content          any             `json:"content"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type OpenAIToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function OpenAIFunctionCall `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object,omitempty"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// ─── Conversion: Anthropic → OpenAI ─────────────────────────

func convertMessages(msgs []MessageParam) []OpenAIMessage {
	var result []OpenAIMessage

	for _, msg := range msgs {
		role := strings.ToLower(msg.Role)
		if len(msg.Content) == 0 {
			result = append(result, OpenAIMessage{Role: role, Content: ""})
			continue
		}

		switch role {
		case "assistant":
			var textParts []string
			var toolCalls []OpenAIToolCall

			for _, block := range msg.Content {
				switch block.Type {
				case "text":
					if block.Text != nil {
						textParts = append(textParts, *block.Text)
					}
				case "thinking":
					if block.Thinking != nil && *block.Thinking != "" {
						textParts = append(textParts, *block.Thinking)
					}
				case "tool_use":
					args := "{}"
					if len(block.Input) > 0 {
						args = string(block.Input)
					}
					id := block.ID
					if id == "" {
						id = generateID("call")
					}
					toolCalls = append(toolCalls, OpenAIToolCall{
						ID:   id,
						Type: "function",
						Function: OpenAIFunctionCall{
							Name:      block.Name,
							Arguments: args,
						},
					})
				}
			}

			m := OpenAIMessage{Role: "assistant"}
			if len(textParts) > 0 {
				m.Content = strings.Join(textParts, "\n")
			}
			if len(toolCalls) > 0 {
				m.ToolCalls = toolCalls
			}
			result = append(result, m)

		case "user":
			var contentParts []any
			var toolResults []OpenAIMessage
			hasImages := false

			for _, block := range msg.Content {
				switch block.Type {
				case "text":
					if block.Text != nil {
						contentParts = append(contentParts, map[string]any{
							"type": "text",
							"text": *block.Text,
						})
					}
				case "image":
					if block.Source != nil && block.Source.Type == "base64" {
						hasImages = true
						dataURL := fmt.Sprintf("data:%s;base64,%s", block.Source.MediaType, block.Source.Data)
						contentParts = append(contentParts, map[string]any{
							"type":      "image_url",
							"image_url": map[string]string{"url": dataURL},
						})
					}
				case "tool_result":
					rc := extractToolResultContent(block.Content)
					toolResults = append(toolResults, OpenAIMessage{
						Role:       "tool",
						Content:    rc,
						ToolCallID: block.ToolUseID,
					})
				}
			}

			// Tool results as separate messages
			result = append(result, toolResults...)

			if len(contentParts) > 0 {
				if hasImages {
					// Mixed content: use array format for vision
					result = append(result, OpenAIMessage{Role: "user", Content: contentParts})
				} else {
					// Text only: use string format for broad compatibility
					var texts []string
					for _, p := range contentParts {
						if m, ok := p.(map[string]any); ok {
							if t, ok := m["text"].(string); ok {
								texts = append(texts, t)
							}
						}
					}
					result = append(result, OpenAIMessage{Role: "user", Content: strings.Join(texts, "\n")})
				}
			} else if len(toolResults) == 0 {
				result = append(result, OpenAIMessage{Role: "user", Content: ""})
			}

		default:
			var textParts []string
			for _, block := range msg.Content {
				if block.Type == "text" && block.Text != nil {
					textParts = append(textParts, *block.Text)
				}
			}
			result = append(result, OpenAIMessage{Role: role, Content: strings.Join(textParts, "\n")})
		}
	}
	return result
}

func extractToolResultContent(content any) string {
	if content == nil {
		return ""
	}
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, item := range c {
			if m, ok := item.(map[string]any); ok {
				if m["type"] == "text" {
					if t, ok := m["text"].(string); ok {
						parts = append(parts, t)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		data, _ := json.Marshal(content)
		return string(data)
	}
}

func convertTools(tools []AnthropicTool) []OpenAITool {
	var result []OpenAITool
	for _, t := range tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		result = append(result, OpenAITool{
			Type: "function",
			Function: OpenAIFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return result
}

func convertToolChoice(tc *ToolChoice) any {
	if tc == nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "tool":
		return map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
	case "none":
		return "none"
	}
	return "auto"
}

func buildOpenAIRequest(req MessagesRequest, model string) OpenAIRequest {
	var messages []OpenAIMessage

	// System prompt
	if req.System != nil {
		switch sys := req.System.(type) {
		case string:
			if sys != "" {
				messages = append(messages, OpenAIMessage{Role: "system", Content: sys})
			}
		case []any:
			var parts []string
			for _, block := range sys {
				if m, ok := block.(map[string]any); ok {
					if m["type"] == "text" {
						if t, ok := m["text"].(string); ok {
							parts = append(parts, t)
						}
					}
				}
			}
			if len(parts) > 0 {
				messages = append(messages, OpenAIMessage{Role: "system", Content: strings.Join(parts, "\n")})
			}
		}
	}

	messages = append(messages, convertMessages(req.Messages)...)

	r := OpenAIRequest{
		Model:    model,
		Messages: messages,
		Stream:   req.Stream,
	}

	if req.Stream {
		r.StreamOptions = &OpenAIStreamOpts{IncludeUsage: true}
	}

	if req.MaxTokens > 0 {
		r.MaxTokens = req.MaxTokens
	}
	if req.Temperature != nil {
		r.Temperature = req.Temperature
	}
	if req.TopP != nil {
		r.TopP = req.TopP
	}
	if req.TopK != nil {
		r.TopK = req.TopK
	}
	if len(req.StopSequences) > 0 {
		r.Stop = req.StopSequences
	}
	if len(req.Tools) > 0 {
		r.Tools = convertTools(req.Tools)
		tc := convertToolChoice(req.ToolChoice)
		if tc != nil {
			r.ToolChoice = tc
		}
	}
	return r
}

// ─── Conversion: OpenAI → Anthropic ─────────────────────────

func mapStopReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func mapErrorType(statusCode int) string {
	switch statusCode {
	case 400:
		return "invalid_request_error"
	case 401:
		return "authentication_error"
	case 403:
		return "permission_error"
	case 404:
		return "not_found_error"
	case 429:
		return "rate_limit_error"
	case 503, 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

func buildAnthropicResponse(oaiResp OpenAIResponse, requestModel string) AnthropicResponse {
	var content []ContentBlock
	var choice OpenAIChoice
	if len(oaiResp.Choices) > 0 {
		choice = oaiResp.Choices[0]
	}

	msg := choice.Message

	// Thinking/reasoning content (before text, per Anthropic convention)
	if msg.ReasoningContent != nil && *msg.ReasoningContent != "" {
		content = append(content, ContentBlock{Type: "thinking", Thinking: msg.ReasoningContent})
	}

	// Text content
	if msg.Content != nil {
		if text, ok := msg.Content.(string); ok && text != "" {
			content = append(content, ContentBlock{Type: "text", Text: ptr(text)})
		}
	}

	// Tool calls
	for _, tc := range msg.ToolCalls {
		content = append(content, ContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}

	if len(content) == 0 {
		content = append(content, ContentBlock{Type: "text", Text: ptr("")})
	}

	var usage AnthropicUsage
	if oaiResp.Usage != nil {
		usage.InputTokens = oaiResp.Usage.PromptTokens
		usage.OutputTokens = oaiResp.Usage.CompletionTokens
	}

	return AnthropicResponse{
		ID:         "msg_" + oaiResp.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      requestModel,
		Content:    content,
		StopReason: mapStopReason(choice.FinishReason, len(msg.ToolCalls) > 0),
		Usage:      usage,
	}
}

// ─── Streaming: OpenAI SSE → Anthropic SSE ──────────────────

type StreamConverter struct {
	model           string
	id              string
	contentIndex    int
	textStarted     bool
	thinkingStarted bool
	thinkingDone    bool
	toolCallsAcc    map[int]*toolCallAcc
	toolCallsSent   map[string]bool
	inputTokens     int
	outputTokens    int
	started         bool
	finished        bool
}

type toolCallAcc struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func newStreamConverter(model string) *StreamConverter {
	return &StreamConverter{
		model:         model,
		id:            generateID("msg"),
		toolCallsAcc:  make(map[int]*toolCallAcc),
		toolCallsSent: make(map[string]bool),
	}
}

func writeSSE(w http.ResponseWriter, eventType string, data any) {
	d, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, d)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func (sc *StreamConverter) start(w http.ResponseWriter) {
	if sc.started {
		return
	}
	sc.started = true
	writeSSE(w, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":      sc.id,
			"type":    "message",
			"role":    "assistant",
			"model":   sc.model,
			"content": []any{},
			"usage":   map[string]int{"input_tokens": 0, "output_tokens": 0},
		},
	})
}

func (sc *StreamConverter) closeThinking(w http.ResponseWriter) {
	if sc.thinkingStarted && !sc.thinkingDone {
		writeSSE(w, "content_block_stop", map[string]any{
			"type": "content_block_stop", "index": sc.contentIndex,
		})
		sc.contentIndex++
		sc.thinkingStarted = false
		sc.thinkingDone = true
	}
}

func (sc *StreamConverter) closeText(w http.ResponseWriter) {
	if sc.textStarted {
		writeSSE(w, "content_block_stop", map[string]any{
			"type": "content_block_stop", "index": sc.contentIndex,
		})
		sc.contentIndex++
		sc.textStarted = false
	}
}

func (sc *StreamConverter) closeOpenBlocks(w http.ResponseWriter) {
	sc.closeThinking(w)
	sc.closeText(w)
}

func (sc *StreamConverter) processChunk(w http.ResponseWriter, chunk OpenAIResponse) {
	sc.start(w)

	if len(chunk.Choices) == 0 {
		if chunk.Usage != nil {
			sc.inputTokens = chunk.Usage.PromptTokens
			sc.outputTokens = chunk.Usage.CompletionTokens
		}
		return
	}

	delta := chunk.Choices[0].Delta

	// Reasoning/thinking content (reasoning_content field from DeepSeek, QwQ, etc.)
	if delta.ReasoningContent != nil && *delta.ReasoningContent != "" {
		if !sc.thinkingStarted {
			sc.thinkingStarted = true
			writeSSE(w, "content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         sc.contentIndex,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
		}
		writeSSE(w, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": sc.contentIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": *delta.ReasoningContent},
		})
	}

	// Text content
	if delta.Content != nil {
		if text, ok := delta.Content.(string); ok && text != "" {
			// Transition: close thinking block before starting text
			sc.closeThinking(w)

			if !sc.textStarted {
				sc.textStarted = true
				writeSSE(w, "content_block_start", map[string]any{
					"type":          "content_block_start",
					"index":         sc.contentIndex,
					"content_block": map[string]any{"type": "text", "text": ""},
				})
			}
			writeSSE(w, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": sc.contentIndex,
				"delta": map[string]any{"type": "text_delta", "text": text},
			})
		}
	}

	// Tool calls in stream
	for _, tc := range delta.ToolCalls {
		idx := 0
		if tc.Index != nil {
			idx = *tc.Index
		}

		if tc.ID != "" {
			if _, exists := sc.toolCallsAcc[idx]; !exists {
				sc.toolCallsAcc[idx] = &toolCallAcc{
					ID:   tc.ID,
					Name: tc.Function.Name,
				}
			}
		}
		if acc, exists := sc.toolCallsAcc[idx]; exists {
			acc.Arguments.WriteString(tc.Function.Arguments)
		}
	}

	// Usage
	if chunk.Usage != nil {
		sc.inputTokens = chunk.Usage.PromptTokens
		sc.outputTokens = chunk.Usage.CompletionTokens
	}

	// Finish
	if chunk.Choices[0].FinishReason != "" && !sc.finished {
		sc.finished = true
		sc.finish(w, chunk.Choices[0].FinishReason)
	}
}

func (sc *StreamConverter) finish(w http.ResponseWriter, finishReason string) {
	sc.closeOpenBlocks(w)

	// Emit accumulated tool calls
	hasToolCalls := len(sc.toolCallsAcc) > 0
	indices := make([]int, 0, len(sc.toolCallsAcc))
	for idx := range sc.toolCallsAcc {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	for _, idx := range indices {
		tc := sc.toolCallsAcc[idx]
		if sc.toolCallsSent[tc.ID] {
			continue
		}

		writeSSE(w, "content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": sc.contentIndex,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Name,
				"input": map[string]any{},
			},
		})
		writeSSE(w, "content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": sc.contentIndex,
			"delta": map[string]any{
				"type":         "input_json_delta",
				"partial_json": tc.Arguments.String(),
			},
		})
		writeSSE(w, "content_block_stop", map[string]any{
			"type": "content_block_stop", "index": sc.contentIndex,
		})
		sc.toolCallsSent[tc.ID] = true
		sc.contentIndex++
	}

	// Empty response fallback
	if sc.contentIndex == 0 {
		writeSSE(w, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		writeSSE(w, "content_block_stop", map[string]any{
			"type": "content_block_stop", "index": 0,
		})
	}

	stopReason := mapStopReason(finishReason, hasToolCalls)
	writeSSE(w, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason},
		"usage": map[string]int{"output_tokens": sc.outputTokens},
	})
	writeSSE(w, "message_stop", map[string]any{"type": "message_stop"})
}

// ─── HTTP handlers ──────────────────────────────────────────

func handleMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeErrorResponse(w, 400, "invalid_request_error", "Failed to read request body")
		return
	}

	var req MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeErrorResponse(w, 400, "invalid_request_error", "Invalid JSON: "+err.Error())
		return
	}

	// Route: configured provider or passthrough to Anthropic
	provider, upstreamModel := resolveModel(req.Model)
	if provider == nil {
		passthroughAnthropic(w, r, body)
		return
	}

	oaiReq := buildOpenAIRequest(req, upstreamModel)
	log.Printf("[proxy] → %s | model=%s | %d msgs | %d tools | stream=%v",
		provider.URL, upstreamModel, len(oaiReq.Messages), len(oaiReq.Tools), req.Stream)

	payload, _ := json.Marshal(oaiReq)
	upstream, err := http.NewRequest("POST", provider.URL, bytes.NewReader(payload))
	if err != nil {
		writeErrorResponse(w, 502, "api_error", "Failed to create upstream request")
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Authorization", "Bearer "+provider.apiKey)

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(upstream)
	if err != nil {
		log.Printf("[proxy] connection error: %v", err)
		writeErrorResponse(w, 502, "api_error", "Upstream connection failed: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("[proxy] upstream error %d: %s", resp.StatusCode, truncate(string(respBody), 200))
		writeErrorResponse(w, resp.StatusCode, mapErrorType(resp.StatusCode),
			fmt.Sprintf("Upstream error: %s", truncate(string(respBody), 500)))
		return
	}

	if req.Stream {
		handleStream(w, resp, req.Model)
	} else {
		handleNonStream(w, resp, req.Model)
	}
}

func handleNonStream(w http.ResponseWriter, resp *http.Response, requestModel string) {
	var oaiResp OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		writeErrorResponse(w, 502, "api_error", "Failed to parse upstream response")
		return
	}

	anthropicResp := buildAnthropicResponse(oaiResp, requestModel)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(anthropicResp)
}

func handleStream(w http.ResponseWriter, resp *http.Response, requestModel string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	converter := newStreamConverter(requestModel)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			if !converter.finished {
				converter.finished = true
				converter.finish(w, "stop")
			}
			break
		}

		var chunk OpenAIResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		converter.processChunk(w, chunk)
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	type modelEntry struct {
		ID       string `json:"id"`
		Provider string `json:"provider"`
		Upstream string `json:"upstream_model"`
	}

	var models []modelEntry
	for alias, route := range cfg.Models {
		provName, upModel, _ := splitRoute(route)
		models = append(models, modelEntry{
			ID:       alias,
			Provider: provName,
			Upstream: upModel,
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   models,
	})
}

func passthroughAnthropic(w http.ResponseWriter, r *http.Request, body []byte) {
	url := anthropicAPI + r.URL.Path

	upstream, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		writeErrorResponse(w, 502, "api_error", "Failed to create upstream request")
		return
	}

	// Forward all Anthropic headers
	for _, header := range []string{"Content-Type", "Anthropic-Version", "X-Api-Key", "Authorization", "Anthropic-Beta"} {
		if v := r.Header.Get(header); v != "" {
			upstream.Header.Set(header, v)
		}
	}

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(upstream)
	if err != nil {
		writeErrorResponse(w, 502, "api_error", "Anthropic API connection failed")
		return
	}
	defer resp.Body.Close()

	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func writeErrorResponse(w http.ResponseWriter, code int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(AnthropicError{
		Type: "error",
		Error: struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		}{Type: errType, Message: message},
	})
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ─── Server ─────────────────────────────────────────────────

func startServer() {
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/messages", handleMessages)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	// Catch-all: passthrough to Anthropic
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		passthroughAnthropic(w, r, body)
	})

	addr := fmt.Sprintf(":%d", cfg.Port)
	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 10 * time.Minute,
		IdleTimeout:  120 * time.Second,
	}
	log.Fatal(server.ListenAndServe())
}

func waitForServer() bool {
	client := &http.Client{Timeout: 500 * time.Millisecond}
	for i := 0; i < 20; i++ {
		resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", cfg.Port))
		if err == nil {
			resp.Body.Close()
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

func findClaude() string {
	paths := []string{
		"claude",
		"/opt/homebrew/bin/claude",
		"/usr/local/bin/claude",
	}
	matches, _ := filepath.Glob("/opt/homebrew/Caskroom/claude-code/*/claude")
	paths = append(paths, matches...)

	for _, p := range paths {
		if path, err := exec.LookPath(p); err == nil {
			return path
		}
	}
	return ""
}

// ─── CLI ────────────────────────────────────────────────────

func printUsage() {
	fmt.Println("openClaude — Use any OpenAI-compatible model with Claude Code")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  openClaude --model <model>       Start proxy + Claude Code")
	fmt.Println("  openClaude serve                 Start proxy server only")
	fmt.Println("  openClaude list                  List configured models")
	fmt.Println()
	fmt.Println("Config:")
	fmt.Println("  Place config.json next to the binary or set OCC_CONFIG env var.")
	fmt.Println("  See config.example.json for the format.")
	fmt.Println()
	fmt.Println("Environment:")
	fmt.Println("  OCC_CONFIG    Path to config file (default: ./config.json)")
	fmt.Println("  PORT          Override listen port from config")
}

func findModel(args []string) string {
	for i, a := range args {
		if a == "--model" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func launch(model string) {
	if p, _ := resolveModel(model); p == nil {
		fmt.Printf("Error: unknown model %q\n", model)
		fmt.Println()
		fmt.Println("Available models:")
		for _, k := range sortedKeys(cfg.Models) {
			fmt.Printf("  %s\n", k)
		}
		os.Exit(1)
	}

	claudePath := findClaude()
	if claudePath == "" {
		log.Fatal("Could not find 'claude' binary. Install Claude Code first.")
	}

	log.Printf("Starting proxy on port %d...", cfg.Port)
	go startServer()

	if !waitForServer() {
		log.Fatal("Proxy failed to start")
	}
	log.Printf("Proxy ready. Launching Claude Code with model %s...", model)
	fmt.Println()

	cmd := exec.Command(claudePath, "--model", model)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("ANTHROPIC_BASE_URL=http://localhost:%d", cfg.Port),
	)
	if err := cmd.Run(); err != nil {
		os.Exit(1)
	}
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		printUsage()
		os.Exit(0)
	}

	// `openClaude --model qwen3-max` — default action is launch
	if model := findModel(args); model != "" {
		loadConfig()
		launch(model)
		return
	}

	switch args[0] {
	case "serve":
		loadConfig()
		log.Printf("Proxy listening on port %d", cfg.Port)
		log.Println("  Unrecognized models → Anthropic API (passthrough)")
		for _, k := range sortedKeys(cfg.Models) {
			provName, upModel, _ := splitRoute(cfg.Models[k])
			log.Printf("  %-24s → %s (%s)", k, upModel, provName)
		}
		fmt.Println()
		startServer()

	case "list":
		loadConfig()
		fmt.Println("Configured models:")
		for _, k := range sortedKeys(cfg.Models) {
			provName, upModel, _ := splitRoute(cfg.Models[k])
			fmt.Printf("  %-24s → %s (%s)\n", k, upModel, provName)
		}

	default:
		printUsage()
		os.Exit(1)
	}
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
