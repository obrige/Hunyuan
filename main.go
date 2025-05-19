package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type WorkerPool struct {
	taskQueue       chan *Task
	workerCount     int
	shutdownChannel chan struct{}
	wg              sync.WaitGroup
}

type Task struct {
	r         *http.Request
	w         http.ResponseWriter
	done      chan struct{}
	reqID     string
	isStream  bool
	hunyuanReq HunyuanRequest
}

func NewWorkerPool(workerCount int, queueSize int) *WorkerPool {
	pool := &WorkerPool{
		taskQueue:       make(chan *Task, queueSize),
		workerCount:     workerCount,
		shutdownChannel: make(chan struct{}),
	}

	pool.Start()
	return pool
}

func (pool *WorkerPool) Start() {
	for i := 0; i < pool.workerCount; i++ {
		pool.wg.Add(1)
		go func(workerID int) {
			defer pool.wg.Done()

			logInfo("Worker %d 已启动", workerID)

			for {
				select {
				case task, ok := <-pool.taskQueue:
					if !ok {
						logInfo("Worker %d 收到队列关闭信号，准备退出", workerID)
						return
					}

					logDebug("Worker %d 处理任务 reqID:%s", workerID, task.reqID)

					if task.isStream {
						err := handleStreamingRequest(task.w, task.r, task.hunyuanReq, task.reqID, getModelCapabilities(task.hunyuanReq.Model))
						if err != nil {
							logError("Worker %d 处理流式任务失败: %v", workerID, err)
						}
					} else {
						err := handleNonStreamingRequest(task.w, task.r, task.hunyuanReq, task.reqID, getModelCapabilities(task.hunyuanReq.Model))
						if err != nil {
							logError("Worker %d 处理非流式任务失败: %v", workerID, err)
						}
					}

					close(task.done)

				case <-pool.shutdownChannel:
					logInfo("Worker %d 收到关闭信号，准备退出", workerID)
					return
				}
			}
		}(i)
	}
}

func (pool *WorkerPool) SubmitTask(task *Task) (bool, error) {
	select {
	case pool.taskQueue <- task:
		return true, nil
	default:
		return false, fmt.Errorf("任务队列已满")
	}
}

func (pool *WorkerPool) Shutdown() {
	logInfo("正在关闭工作池...")

	close(pool.shutdownChannel)

	pool.wg.Wait()

	close(pool.taskQueue)

	logInfo("工作池已关闭")
}

type Semaphore struct {
	sem chan struct{}
}

func NewSemaphore(size int) *Semaphore {
	return &Semaphore{
		sem: make(chan struct{}, size),
	}
}

func (s *Semaphore) Acquire() {
	s.sem <- struct{}{}
}

func (s *Semaphore) Release() {
	<-s.sem
}

func (s *Semaphore) TryAcquire() bool {
	select {
	case s.sem <- struct{}{}:
		return true
	default:
		return false
	}
}

type Config struct {
	Port             string
	Address          string
	LogLevel         string
	DevMode          bool
	MaxRetries       int
	Timeout          int
	VerifySSL        bool
	ModelName        string
	BearerToken      string
	WorkerCount      int
	QueueSize        int
	MaxConcurrent    int
}

var SupportedModels = []string{
	"hunyuan-t1-latest",
	"hunyuan-turbos-latest",
}

var modelCapabilities = map[string]map[string]bool{
	"hunyuan-t1-latest": {
		"chat":        true,
		"completions": true,
		"reasoning":   true,
	},
	"hunyuan-turbos-latest": {
		"chat":        true,
		"completions": true,
		"reasoning":   false,
	},
}

func getModelCapabilities(modelName string) map[string]bool {
	if capabilities, ok := modelCapabilities[modelName]; ok {
		return capabilities
	}
	return map[string]bool{
		"chat":        false,
		"completions": false,
		"reasoning":   false,
	}
}

const (
	TargetURL = "https://llm.hunyuan.tencent.com/aide/api/v2/triton_image/demo_text_chat/"
	Version   = "1.0.0"
)

const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"
)

func parseFlags() *Config {
	cfg := &Config{}
	flag.StringVar(&cfg.Port, "port", "6666", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "0.0.0.0", "Address to listen on")
	flag.StringVar(&cfg.LogLevel, "log-level", LogLevelInfo, "Log level (debug, info, warn, error)")
	flag.BoolVar(&cfg.DevMode, "dev", false, "Enable development mode with enhanced logging")
	flag.IntVar(&cfg.MaxRetries, "max-retries", 3, "Maximum number of retries for failed requests")
	flag.IntVar(&cfg.Timeout, "timeout", 300, "Request timeout in seconds")
	flag.BoolVar(&cfg.VerifySSL, "verify-ssl", true, "Verify SSL certificates")
	flag.StringVar(&cfg.ModelName, "model", "hunyuan-t1-latest", "Default Hunyuan model name")
	flag.StringVar(&cfg.BearerToken, "token", "7auGXNATFSKl7dF", "Bearer token for Hunyuan API")
	flag.IntVar(&cfg.WorkerCount, "workers", 50, "Number of worker goroutines in the pool")
	flag.IntVar(&cfg.QueueSize, "queue-size", 500, "Size of the task queue")
	flag.IntVar(&cfg.MaxConcurrent, "max-concurrent", 100, "Maximum number of concurrent requests")
	flag.Parse()

	if cfg.DevMode && cfg.LogLevel != LogLevelDebug {
		cfg.LogLevel = LogLevelDebug
		fmt.Println("开发模式已启用，日志级别设置为debug")
	}

	return cfg
}

var (
	appConfig *Config
)

var (
	requestCounter    int64
	successCounter    int64
	errorCounter      int64
	avgResponseTime   int64
	latencyHistogram  [10]int64
	queuedRequests    int64
	rejectedRequests  int64
)

var (
	workerPool     *WorkerPool
	requestSem     *Semaphore
)

var (
	logger    *log.Logger
	logLevel  string
	logMutex  sync.Mutex
)

func initLogger(level string) {
	logger = log.New(os.Stdout, "[HunyuanAPI] ", log.LstdFlags)
	logLevel = level
}

func logDebug(format string, v ...interface{}) {
	if logLevel == LogLevelDebug {
		logMutex.Lock()
		logger.Printf("[DEBUG] "+format, v...)
		logMutex.Unlock()
	}
}

func logInfo(format string, v ...interface{}) {
	if logLevel == LogLevelDebug || logLevel == LogLevelInfo {
		logMutex.Lock()
		logger.Printf("[INFO] "+format, v...)
		logMutex.Unlock()
	}
}

func logWarn(format string, v ...interface{}) {
	if logLevel == LogLevelDebug || logLevel == LogLevelInfo || logLevel == LogLevelWarn {
		logMutex.Lock()
		logger.Printf("[WARN] "+format, v...)
		logMutex.Unlock()
	}
}

func logError(format string, v ...interface{}) {
	logMutex.Lock()
	logger.Printf("[ERROR] "+format, v...)
	logMutex.Unlock()

	atomic.AddInt64(&errorCounter, 1)
}

type APIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type APIRequest struct {
	Model       string       `json:"model"`
	Messages    []APIMessage `json:"messages"`
	Stream      bool         `json:"stream"`
	Temperature float64      `json:"temperature,omitempty"`
	MaxTokens   int          `json:"max_tokens,omitempty"`
}

type HunyuanRequest struct {
	Stream           bool         `json:"stream"`
	Model            string       `json:"model"`
	QueryID          string       `json:"query_id"`
	Messages         []APIMessage `json:"messages"`
	StreamModeration bool         `json:"stream_moderation"`
	EnableEnhancement bool        `json:"enable_enhancement"`
}

type HunyuanResponse struct {
	ID                string      `json:"id"`
	Object            string      `json:"object"`
	Created           int64       `json:"created"`
	Model             string      `json:"model"`
	SystemFingerprint string      `json:"system_fingerprint"`
	Choices           []Choice    `json:"choices"`
	Note              string      `json:"note,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type Delta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type StreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		FinishReason *string `json:"finish_reason,omitempty"`
		Delta        struct {
			Role             string `json:"role,omitempty"`
			Content          string `json:"content,omitempty"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta"`
	} `json:"choices"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

var (
	requestCount uint64 = 0
	countMutex   sync.Mutex
)

func main() {
	appConfig = parseFlags()

	initLogger(appConfig.LogLevel)

	logInfo("启动服务: TargetURL=%s, Address=%s, Port=%s, Version=%s, LogLevel=%s, 支持模型=%v, BearerToken=***, WorkerCount=%d, QueueSize=%d, MaxConcurrent=%d",
		TargetURL, appConfig.Address, appConfig.Port, Version, appConfig.LogLevel, SupportedModels,
		appConfig.WorkerCount, appConfig.QueueSize, appConfig.MaxConcurrent)

	workerPool = NewWorkerPool(appConfig.WorkerCount, appConfig.QueueSize)
	requestSem = NewSemaphore(appConfig.MaxConcurrent)

	logInfo("工作池已创建: %d个worker, 队列大小为%d", appConfig.WorkerCount, appConfig.QueueSize)

	http.DefaultTransport.(*http.Transport).MaxIdleConnsPerHost = 100
	http.DefaultTransport.(*http.Transport).MaxIdleConns = 100
	http.DefaultTransport.(*http.Transport).IdleConnTimeout = 90 * time.Second

	server := &http.Server{
		Addr:         appConfig.Address + ":" + appConfig.Port,
		ReadTimeout:  time.Duration(appConfig.Timeout) * time.Second,
		WriteTimeout: time.Duration(appConfig.Timeout) * time.Second,
		IdleTimeout:  120 * time.Second,
		Handler:      nil,
	}

	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		handleModelsRequest(w, r)
	})

	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		countMutex.Lock()
		requestCount++
		currentCount := requestCount
		countMutex.Unlock()

		logInfo("收到新请求 #%d", currentCount)

		atomic.AddInt64(&requestCounter, 1)

		if !requestSem.TryAcquire() {
			atomic.AddInt64(&rejectedRequests, 1)
			logWarn("请求 #%d 被拒绝: 当前并发请求数已达上限", currentCount)
			w.Header().Set("Retry-After", "30")
			http.Error(w, "Server is busy, please try again later", http.StatusServiceUnavailable)
			return
		}

		defer requestSem.Release()

		handleChatCompletionRequestWithPool(w, r, currentCount)
	})

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		reqCount := atomic.LoadInt64(&requestCounter)
		succCount := atomic.LoadInt64(&successCounter)
		errCount := atomic.LoadInt64(&errorCounter)
		queuedCount := atomic.LoadInt64(&queuedRequests)
		rejectedCount := atomic.LoadInt64(&rejectedRequests)

		var avgTime int64 = 0
		if reqCount > 0 {
			avgTime = atomic.LoadInt64(&avgResponseTime) / max(reqCount, 1)
		}

		histogram := make([]int64, 10)
		for i := 0; i < 10; i++ {
			histogram[i] = atomic.LoadInt64(&latencyHistogram[i])
		}

		stats := map[string]interface{}{
			"status":           "ok",
			"version":          Version,
			"requests":         reqCount,
			"success":          succCount,
			"errors":           errCount,
			"queued":           queuedCount,
			"rejected":         rejectedCount,
			"avg_time_ms":      avgTime,
			"histogram_ms":     histogram,
			"worker_count":     workerPool.workerCount,
			"queue_size":       len(workerPool.taskQueue),
			"queue_capacity":   cap(workerPool.taskQueue),
			"queue_percent":    float64(len(workerPool.taskQueue)) / float64(cap(workerPool.taskQueue)) * 100,
			"concurrent_limit": appConfig.MaxConcurrent,
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(stats)
	})

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		logInfo("Starting proxy server on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logError("Failed to start server: %v", err)
			os.Exit(1)
		}
	}()

	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	logInfo("Server is shutting down...")
	if err := server.Shutdown(ctx); err != nil {
		logError("Server shutdown failed: %v", err)
	}

	workerPool.Shutdown()

	logInfo("Server gracefully stopped")
}

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

func validateMessages(messages []APIMessage) (bool, string) {
	reqID := generateRequestID()
	logDebug("[reqID:%s] 验证消息格式", reqID)

	if messages == nil || len(messages) == 0 {
		return false, "Messages array is required"
	}

	for _, msg := range messages {
		if msg.Role == "" || msg.Content == nil {
			return false, "Invalid message format: each message must have role and content"
		}
	}

	return true, ""
}

func extractToken(r *http.Request) (string, error) {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return "", fmt.Errorf("missing Authorization header")
	}

	if !strings.HasPrefix(authHeader, "Bearer ") {
		return "", fmt.Errorf("invalid Authorization header format, must start with 'Bearer '")
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == "" {
		return "", fmt.Errorf("empty token in Authorization header")
	}

	return token, nil
}

func contentToString(content interface{}) string {
	if content == nil {
		return ""
	}

	switch v := content.(type) {
	case string:
		return v
	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			logWarn("将内容转换为JSON失败: %v", err)
			return ""
		}
		return string(jsonBytes)
	}
}

func generateQueryID() string {
	return fmt.Sprintf("%d%s", time.Now().UnixNano(), getRandomString(8))
}

func isModelSupported(modelName string) bool {
	_, ok := modelCapabilities[modelName]
	return ok
}

func handleModelsRequest(w http.ResponseWriter, r *http.Request) {
	logInfo("处理模型列表请求")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	modelData := make([]map[string]interface{}, 0, len(modelCapabilities))
	for modelName, capabilities := range modelCapabilities {
		modelData = append(modelData, map[string]interface{}{
			"id":       modelName,
			"object":   "model",
			"created":  time.Now().Unix(),
			"owned_by": "TencentCloud",
			"capabilities": capabilities,
		})
	}

	modelsList := map[string]interface{}{
		"object": "list",
		"data":   modelData,
	}

	json.NewEncoder(w).Encode(modelsList)
}

func handleChatCompletionRequestWithPool(w http.ResponseWriter, r *http.Request, requestNum uint64) {
	reqID := generateRequestID()
	startTime := time.Now()
	logInfo("[reqID:%s] 处理聊天补全请求 #%d", reqID, requestNum)

	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(appConfig.Timeout)*time.Second)
	defer cancel()

	r = r.WithContext(ctx)

	defer func() {
		if r := recover(); r != nil {
			logError("[reqID:%s] 处理请求时发生panic: %v", reqID, r)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
	}()

	var apiReq APIRequest
	if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
		logError("[reqID:%s] 解析请求失败: %v", reqID, err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	valid, errMsg := validateMessages(apiReq.Messages)
	if !valid {
		logError("[reqID:%s] 消息格式验证失败: %s", reqID, errMsg)
		http.Error(w, errMsg, http.StatusBadRequest)
		return
	}

	isStream := apiReq.Stream

	modelName := appConfig.ModelName
	if apiReq.Model != "" {
		if isModelSupported(apiReq.Model) {
			modelName = apiReq.Model
		} else {
			logWarn("[reqID:%s] 请求的模型 %s 不支持，使用默认模型 %s", reqID, apiReq.Model, modelName)
		}
	}

	logInfo("[reqID:%s] 使用模型: %s", reqID, modelName)

	hunyuanReq := HunyuanRequest{
		Stream:            true,
		Model:             modelName,
		QueryID:           generateQueryID(),
		Messages:          apiReq.Messages,
		StreamModeration:  true,
		EnableEnhancement: false,
	}

	task := &Task{
		r:          r,
		w:          w,
		done:       make(chan struct{}),
		reqID:      reqID,
		isStream:   isStream,
		hunyuanReq: hunyuanReq,
	}

	atomic.AddInt64(&queuedRequests, 1)
	submitted, err := workerPool.SubmitTask(task)
	if !submitted {
		atomic.AddInt64(&queuedRequests, -1)
		atomic.AddInt64(&rejectedRequests, 1)
		logError("[reqID:%s] 提交任务失败: %v", reqID, err)
		w.Header().Set("Retry-After", "60")
		http.Error(w, "Server queue is full, please try again later", http.StatusServiceUnavailable)
		return
	}

	logInfo("[reqID:%s] 任务已提交到队列", reqID)

	select {
	case <-task.done:
		logInfo("[reqID:%s] 任务处理完成", reqID)
	case <-r.Context().Done():
		logWarn("[reqID:%s] 请求被取消或超时: %v", reqID, r.Context().Err())
	}

	atomic.AddInt64(&queuedRequests, -1)
	elapsed := time.Since(startTime).Milliseconds()

	bucketIndex := min(int(elapsed/100), 9)
	atomic.AddInt64(&latencyHistogram[bucketIndex], 1)

	atomic.AddInt64(&avgResponseTime, elapsed)

	logInfo("[reqID:%s] 请求生命周期结束，耗时: %dms", reqID, elapsed)
}

func getHTTPClient() *http.Client {
	tr := &http.Transport{
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		TLSClientConfig:     nil,
	}

	if !appConfig.VerifySSL {
		tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}

	return &http.Client{
		Timeout:   time.Duration(appConfig.Timeout) * time.Second,
		Transport: tr,
	}
}

func handleStreamingRequest(w http.ResponseWriter, r *http.Request, hunyuanReq HunyuanRequest, reqID string, capabilities map[string]bool) error {
	logInfo("[reqID:%s] 处理流式请求 (worker)", reqID)

	jsonData, err := json.Marshal(hunyuanReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		return err
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建上游请求失败: %v", reqID, err)
		if r.Context().Err() == context.DeadlineExceeded {
			return fmt.Errorf("请求超时: %v", r.Context().Err())
		}
		return err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Model", hunyuanReq.Model)
	setCommonHeaders(httpReq)

	client := getHTTPClient()

	resp, err := client.Do(httpReq)
	if err != nil {
		logError("[reqID:%s] 发送上游请求失败: %v", reqID, err)
		if r.Context().Err() == context.DeadlineExceeded {
			return fmt.Errorf("请求超时: %v", r.Context().Err())
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)

		bodyBytes, _ := io.ReadAll(resp.Body)
		logError("[reqID:%s] 错误响应内容: %s", reqID, string(bodyBytes))

		return fmt.Errorf("API返回非200状态码: %d, 响应体: %s", resp.StatusCode, string(bodyBytes))
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	respID := fmt.Sprintf("chatcmpl-%s", getRandomString(10))
	createdTime := time.Now().Unix()

	reader := bufio.NewReaderSize(resp.Body, 16384)

	flusher, ok := w.(http.Flusher)
	if !ok {
		logError("[reqID:%s] Streaming not supported by ResponseWriter", reqID)
		return fmt.Errorf("streaming not supported by ResponseWriter")
	}

	roleChunk := createRoleChunk(respID, createdTime, hunyuanReq.Model)
	w.Write([]byte("data: " + string(roleChunk) + "\n\n"))
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			logWarn("[reqID:%s] 请求超时或被客户端取消: %v", reqID, r.Context().Err())
			flusher.Flush()
			return fmt.Errorf("请求超时或被取消: %v", r.Context().Err())
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				logError("[reqID:%s] 读取上游响应出错: %v", reqID, err)
				return err
			}
			break
		}

		lineStr := string(line)
		if strings.HasPrefix(lineStr, "data: ") {
			jsonStr := strings.TrimPrefix(lineStr, "data: ")
			jsonStr = strings.TrimSpace(jsonStr)

			if jsonStr == "[DONE]" {
				logDebug("[reqID:%s] 收到[DONE]消息", reqID)
				w.Write([]byte("data: [DONE]\n\n"))
				flusher.Flush()
				break
			}

			var hunyuanResp HunyuanResponse
			if err := json.Unmarshal([]byte(jsonStr), &hunyuanResp); err != nil {
				logWarn("[reqID:%s] 解析上游JSON失败: %v, data: %s", reqID, err, jsonStr)
				continue
			}

			for _, choice := range hunyuanResp.Choices {
				if choice.Delta.Content != "" {
					contentChunk := createContentChunk(respID, createdTime, hunyuanReq.Model, choice.Delta.Content)
					w.Write([]byte("data: " + string(contentChunk) + "\n\n"))
					flusher.Flush()
				}

				if capabilities["reasoning"] && choice.Delta.ReasoningContent != "" {
					reasoningChunk := createReasoningChunk(respID, createdTime, hunyuanReq.Model, choice.Delta.ReasoningContent)
					w.Write([]byte("data: " + string(reasoningChunk) + "\n\n"))
					flusher.Flush()
				}

				if choice.FinishReason != nil {
					finishReason := *choice.FinishReason
					if finishReason != "" {
						doneChunk := createDoneChunk(respID, createdTime, hunyuanReq.Model, finishReason)
						w.Write([]byte("data: " + string(doneChunk) + "\n\n"))
						flusher.Flush()
					}
				}
			}
		}
	}

	logDebug("[reqID:%s] 流式响应结束，发送[DONE]", reqID)
	w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()

	return nil
}

func handleNonStreamingRequest(w http.ResponseWriter, r *http.Request, hunyuanReq HunyuanRequest, reqID string, capabilities map[string]bool) error {
	logInfo("[reqID:%s] 处理非流式请求 (worker)", reqID)

	hunyuanReq.Stream = true

	jsonData, err := json.Marshal(hunyuanReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		return err
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建上游请求失败: %v", reqID, err)
		if r.Context().Err() == context.DeadlineExceeded {
			return fmt.Errorf("请求超时: %v", r.Context().Err())
		}
		return err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Model", hunyuanReq.Model)
	setCommonHeaders(httpReq)

	client := getHTTPClient()

	resp, err := client.Do(httpReq)
	if err != nil {
		logError("[reqID:%s] 发送上游请求失败: %v", reqID, err)
		if r.Context().Err() == context.DeadlineExceeded {
			return fmt.Errorf("请求超时: %v", r.Context().Err())
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)

		bodyBytes, _ := io.ReadAll(resp.Body)
		logError("[reqID:%s] 错误响应内容: %s", reqID, string(bodyBytes))

		return fmt.Errorf("API返回非200状态码: %d, 响应体: %s", resp.StatusCode, string(bodyBytes))
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		logError("[reqID:%s] 读取上游响应失败: %v", reqID, err)
		return err
	}

	fullContent, reasoningContent, err := extractFullContentFromStream(bodyBytes, reqID)
	if err != nil {
		logError("[reqID:%s] 解析流式响应失败: %v", reqID, err)
		return err
	}

	completionResponse := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", getRandomString(10)),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   hunyuanReq.Model,
		Choices: []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"message"`
		}{
			{
				Index:        0,
				FinishReason: "stop",
				Message: struct {
					Role             string `json:"role"`
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Role:             "assistant",
					Content:          fullContent,
					ReasoningContent: func() string {
						if capabilities["reasoning"] {
							return reasoningContent
						}
						return ""
					}(),
				},
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     len(formatMessages(hunyuanReq.Messages)) / 4,
			CompletionTokens: len(fullContent) / 4,
			TotalTokens:      (len(formatMessages(hunyuanReq.Messages)) + len(fullContent)) / 4,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(completionResponse); err != nil {
		logError("[reqID:%s] 编码响应失败: %v", reqID, err)
		return err
	}

	return nil
}

func extractFullContentFromStream(bodyBytes []byte, reqID string) (string, string, error) {
	bodyStr := string(bodyBytes)
	lines := strings.Split(bodyStr, "\n")

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			jsonStr := strings.TrimPrefix(line, "data: ")
			jsonStr = strings.TrimSpace(jsonStr)

			var hunyuanResp HunyuanResponse
			if err := json.Unmarshal([]byte(jsonStr), &hunyuanResp); err != nil {
				continue
			}

			for _, choice := range hunyuanResp.Choices {
				if choice.Delta.Content != "" {
					contentBuilder.WriteString(choice.Delta.Content)
				}
				if choice.Delta.ReasoningContent != "" {
					reasoningBuilder.WriteString(choice.Delta.ReasoningContent)
				}
			}
		}
	}

	return contentBuilder.String(), reasoningBuilder.String(), nil
}

func createRoleChunk(id string, created int64, model string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Role: "assistant",
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

func createContentChunk(id string, created int64, model string, content string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Content: content,
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

func createReasoningChunk(id string, created int64, model string, reasoningContent string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					ReasoningContent: reasoningContent,
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

func createDoneChunk(id string, created int64, model string, reason string) []byte {
	finishReason := reason
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index:        0,
				FinishReason: &finishReason,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

func setCommonHeaders(req *http.Request) {
	req.Header.Set("accept", "*/*")
	req.Header.Set("accept-language", "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7")
	req.Header.Set("authorization", "Bearer "+appConfig.BearerToken)
	req.Header.Set("dnt", "1")
	req.Header.Set("origin", "https://llm.hunyuan.tencent.com")
	req.Header.Set("polaris", "stream-server-online-sbs-10697")
	req.Header.Set("priority", "u=1, i")
	req.Header.Set("referer", "https://llm.hunyuan.tencent.com/")
	req.Header.Set("sec-ch-ua", "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"")
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-platform", "\"Windows\"")
	req.Header.Set("sec-fetch-dest", "empty")
	req.Header.Set("sec-fetch-mode", "cors")
	req.Header.Set("sec-fetch-site", "same-origin")
	req.Header.Set("staffname", "staryxzhang")
	req.Header.Set("wsid", "10697")
	req.Header.Set("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
}

func generateRequestID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

func getRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}

func formatMessages(messages []APIMessage) string {
	var result strings.Builder
	for _, msg := range messages {
		result.WriteString(msg.Role)
		result.WriteString(": ")
		result.WriteString(contentToString(msg.Content))
		result.WriteString("\n")
	}
	return result.String()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
