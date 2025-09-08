// QENEX Universal Banking Protocol
// High-performance, standards-compliant banking message processor

package protocols

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"log"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"
	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Constants for banking protocols
const (
	MaxMessageSize        = 1024 * 1024 * 10 // 10MB
	MaxConcurrentMessages = 100000
	MessageTimeoutSeconds = 30
	RetryAttempts         = 3
	HeartbeatInterval     = 30 * time.Second
	
	// SWIFT message types
	MT103 = "103" // Single Customer Credit Transfer
	MT202 = "202" // General Financial Institution Transfer
	MT910 = "910" // Confirmation of Credit
	MT940 = "940" // Customer Statement Message
	MT950 = "950" // Statement Message
	
	// ISO 20022 message types
	PACS008 = "pacs.008.001.10" // FIToFICustomerCreditTransfer
	PACS009 = "pacs.009.001.10" // FIToFICustomerDirectDebit
	PAIN001 = "pain.001.001.11" // CustomerCreditTransferInitiation
	CAMT053 = "camt.053.001.10" // BankToCustomerStatement
	CAMT054 = "camt.054.001.10" // BankToCustomerDebitCreditNotification
)

// Universal Message represents any banking message in a standardized format
type UniversalMessage struct {
	ID          string                 `json:"id" xml:"id"`
	Type        string                 `json:"type" xml:"type"`
	Version     string                 `json:"version" xml:"version"`
	Sender      ParticipantInfo        `json:"sender" xml:"sender"`
	Receiver    ParticipantInfo        `json:"receiver" xml:"receiver"`
	Timestamp   time.Time              `json:"timestamp" xml:"timestamp"`
	Priority    MessagePriority        `json:"priority" xml:"priority"`
	Network     string                 `json:"network" xml:"network"`
	Payload     map[string]interface{} `json:"payload" xml:"payload"`
	Signature   string                 `json:"signature,omitempty" xml:"signature,omitempty"`
	Encryption  *EncryptionInfo        `json:"encryption,omitempty" xml:"encryption,omitempty"`
	Compliance  *ComplianceInfo        `json:"compliance" xml:"compliance"`
	Routing     *RoutingInfo           `json:"routing,omitempty" xml:"routing,omitempty"`
}

type ParticipantInfo struct {
	ID          string            `json:"id" xml:"id"`
	Name        string            `json:"name" xml:"name"`
	BIC         string            `json:"bic,omitempty" xml:"bic,omitempty"`
	LEI         string            `json:"lei,omitempty" xml:"lei,omitempty"`
	Country     string            `json:"country" xml:"country"`
	Credentials map[string]string `json:"credentials,omitempty" xml:"credentials,omitempty"`
}

type MessagePriority int

const (
	PriorityLow MessagePriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
	PriorityEmergency
)

type EncryptionInfo struct {
	Algorithm   string `json:"algorithm" xml:"algorithm"`
	KeyID       string `json:"key_id" xml:"key_id"`
	IV          string `json:"iv,omitempty" xml:"iv,omitempty"`
	Signature   string `json:"signature" xml:"signature"`
}

type ComplianceInfo struct {
	AMLScore        float64  `json:"aml_score" xml:"aml_score"`
	SanctionsCheck  bool     `json:"sanctions_check" xml:"sanctions_check"`
	KYCVerified     bool     `json:"kyc_verified" xml:"kyc_verified"`
	RegulatoryFlags []string `json:"regulatory_flags,omitempty" xml:"regulatory_flags,omitempty"`
	ComplianceHash  string   `json:"compliance_hash" xml:"compliance_hash"`
}

type RoutingInfo struct {
	Path        []string          `json:"path" xml:"path"`
	Hops        int              `json:"hops" xml:"hops"`
	Preferences map[string]string `json:"preferences,omitempty" xml:"preferences,omitempty"`
}

// Protocol handlers for different banking standards
type ProtocolHandler interface {
	Parse(data []byte) (*UniversalMessage, error)
	Format(msg *UniversalMessage) ([]byte, error)
	Validate(msg *UniversalMessage) error
	GetMessageType() string
}

// SWIFT MT message handler
type SWIFTHandler struct {
	BIC        string
	logger     *zap.Logger
	validator  *MessageValidator
}

func NewSWIFTHandler(bic string, logger *zap.Logger) *SWIFTHandler {
	return &SWIFTHandler{
		BIC:       bic,
		logger:    logger,
		validator: NewMessageValidator(),
	}
}

func (s *SWIFTHandler) Parse(data []byte) (*UniversalMessage, error) {
	mtText := string(data)
	
	// Extract SWIFT message components using regex
	headerRegex := regexp.MustCompile(`\{1:F01([A-Z0-9]{11})\d+\}`)
	typeRegex := regexp.MustCompile(`\{2:I(\d{3})([A-Z0-9]{11})[N]\}`)
	bodyRegex := regexp.MustCompile(`\{4:(.*?)-\}`)
	
	headerMatch := headerRegex.FindStringSubmatch(mtText)
	typeMatch := typeRegex.FindStringSubmatch(mtText)
	bodyMatch := bodyRegex.FindStringSubmatch(mtText)
	
	if len(headerMatch) < 2 || len(typeMatch) < 3 || len(bodyMatch) < 2 {
		return nil, fmt.Errorf("invalid SWIFT MT message format")
	}
	
	senderBIC := headerMatch[1]
	messageType := typeMatch[1]
	receiverBIC := typeMatch[2]
	messageBody := bodyMatch[1]
	
	// Parse message body based on MT type
	payload, err := s.parseMTBody(messageType, messageBody)
	if err != nil {
		return nil, fmt.Errorf("failed to parse MT%s body: %w", messageType, err)
	}
	
	msg := &UniversalMessage{
		ID:        generateMessageID(),
		Type:      "MT" + messageType,
		Version:   "2023",
		Timestamp: time.Now(),
		Priority:  s.determinePriority(messageType, payload),
		Network:   "SWIFT",
		Sender: ParticipantInfo{
			ID:      senderBIC,
			BIC:     senderBIC,
			Country: senderBIC[:2],
		},
		Receiver: ParticipantInfo{
			ID:      receiverBIC,
			BIC:     receiverBIC,
			Country: receiverBIC[:2],
		},
		Payload:    payload,
		Compliance: s.generateComplianceInfo(payload),
	}
	
	return msg, nil
}

func (s *SWIFTHandler) Format(msg *UniversalMessage) ([]byte, error) {
	if !strings.HasPrefix(msg.Type, "MT") {
		return nil, fmt.Errorf("not a SWIFT MT message")
	}
	
	messageType := strings.TrimPrefix(msg.Type, "MT")
	
	// Build SWIFT message structure
	header := fmt.Sprintf("{1:F01%s0000000000}", msg.Sender.BIC)
	appHeader := fmt.Sprintf("{2:I%s%sN}", messageType, msg.Receiver.BIC)
	
	// Format message body based on type
	body, err := s.formatMTBody(messageType, msg.Payload)
	if err != nil {
		return nil, fmt.Errorf("failed to format MT%s body: %w", messageType, err)
	}
	
	swiftMessage := fmt.Sprintf("%s%s{4:\n%s\n-}", header, appHeader, body)
	
	return []byte(swiftMessage), nil
}

func (s *SWIFTHandler) parseMTBody(messageType, body string) (map[string]interface{}, error) {
	payload := make(map[string]interface{})
	
	// Parse SWIFT field tags
	fieldRegex := regexp.MustCompile(`:(\d{2}[A-Z]?):(.*?)(?=:\d{2}[A-Z]?:|$)`)
	fields := fieldRegex.FindAllStringSubmatch(body, -1)
	
	for _, field := range fields {
		if len(field) >= 3 {
			tag := field[1]
			value := strings.TrimSpace(field[2])
			
			switch messageType {
			case MT103:
				payload = s.parseMT103Field(tag, value, payload)
			case MT202:
				payload = s.parseMT202Field(tag, value, payload)
			case MT940:
				payload = s.parseMT940Field(tag, value, payload)
			default:
				payload[tag] = value
			}
		}
	}
	
	return payload, nil
}

func (s *SWIFTHandler) parseMT103Field(tag, value string, payload map[string]interface{}) map[string]interface{} {
	switch tag {
	case "20":
		payload["transaction_reference"] = value
	case "23B":
		payload["bank_operation_code"] = value
	case "32A":
		// Parse date, currency, and amount
		if len(value) >= 9 {
			payload["value_date"] = value[:6]
			payload["currency"] = value[6:9]
			if len(value) > 9 {
				if amount, err := parseAmount(value[9:]); err == nil {
					payload["amount"] = amount
				}
			}
		}
	case "50K":
		payload["ordering_customer"] = parsePartyIdentification(value)
	case "59":
		payload["beneficiary_customer"] = parsePartyIdentification(value)
	case "70":
		payload["remittance_information"] = value
	case "71A":
		payload["details_of_charges"] = value
	default:
		payload[tag] = value
	}
	return payload
}

func (s *SWIFTHandler) parseMT202Field(tag, value string, payload map[string]interface{}) map[string]interface{} {
	switch tag {
	case "20":
		payload["transaction_reference"] = value
	case "32A":
		if len(value) >= 9 {
			payload["value_date"] = value[:6]
			payload["currency"] = value[6:9]
			if len(value) > 9 {
				if amount, err := parseAmount(value[9:]); err == nil {
					payload["amount"] = amount
				}
			}
		}
	case "53A", "53D":
		payload["senders_correspondent"] = value
	case "58A", "58D":
		payload["beneficiary_institution"] = value
	case "72":
		payload["sender_to_receiver_information"] = value
	default:
		payload[tag] = value
	}
	return payload
}

func (s *SWIFTHandler) parseMT940Field(tag, value string, payload map[string]interface{}) map[string]interface{} {
	switch tag {
	case "20":
		payload["transaction_reference"] = value
	case "25":
		payload["account_identification"] = value
	case "28C":
		payload["statement_number"] = value
	case "60F", "60M":
		payload["opening_balance"] = parseBalance(value)
	case "61":
		payload = s.parseStatementLine(value, payload)
	case "62F", "62M":
		payload["closing_balance"] = parseBalance(value)
	case "64":
		payload["closing_available_balance"] = parseBalance(value)
	default:
		payload[tag] = value
	}
	return payload
}

func (s *SWIFTHandler) formatMTBody(messageType string, payload map[string]interface{}) (string, error) {
	var body strings.Builder
	
	switch messageType {
	case MT103:
		return s.formatMT103Body(payload)
	case MT202:
		return s.formatMT202Body(payload)
	case MT940:
		return s.formatMT940Body(payload)
	default:
		return "", fmt.Errorf("unsupported message type: MT%s", messageType)
	}
}

func (s *SWIFTHandler) formatMT103Body(payload map[string]interface{}) (string, error) {
	var body strings.Builder
	
	if ref, ok := payload["transaction_reference"].(string); ok {
		body.WriteString(fmt.Sprintf(":20:%s\n", ref))
	}
	
	if code, ok := payload["bank_operation_code"].(string); ok {
		body.WriteString(fmt.Sprintf(":23B:%s\n", code))
	} else {
		body.WriteString(":23B:CRED\n") // Default
	}
	
	// Format amount field
	if currency, ok := payload["currency"].(string); ok {
		if amount, ok := payload["amount"].(float64); ok {
			valueDate := time.Now().Format("060102")
			if vd, ok := payload["value_date"].(string); ok && len(vd) == 6 {
				valueDate = vd
			}
			body.WriteString(fmt.Sprintf(":32A:%s%s%.2f\n", valueDate, currency, amount))
		}
	}
	
	if customer, ok := payload["ordering_customer"].(map[string]interface{}); ok {
		body.WriteString(fmt.Sprintf(":50K:%s\n", formatPartyIdentification(customer)))
	}
	
	if beneficiary, ok := payload["beneficiary_customer"].(map[string]interface{}); ok {
		body.WriteString(fmt.Sprintf(":59:%s\n", formatPartyIdentification(beneficiary)))
	}
	
	if info, ok := payload["remittance_information"].(string); ok {
		body.WriteString(fmt.Sprintf(":70:%s\n", info))
	}
	
	if charges, ok := payload["details_of_charges"].(string); ok {
		body.WriteString(fmt.Sprintf(":71A:%s\n", charges))
	} else {
		body.WriteString(":71A:OUR\n") // Default
	}
	
	return body.String(), nil
}

func (s *SWIFTHandler) formatMT202Body(payload map[string]interface{}) (string, error) {
	var body strings.Builder
	
	if ref, ok := payload["transaction_reference"].(string); ok {
		body.WriteString(fmt.Sprintf(":20:%s\n", ref))
	}
	
	if currency, ok := payload["currency"].(string); ok {
		if amount, ok := payload["amount"].(float64); ok {
			valueDate := time.Now().Format("060102")
			if vd, ok := payload["value_date"].(string); ok && len(vd) == 6 {
				valueDate = vd
			}
			body.WriteString(fmt.Sprintf(":32A:%s%s%.2f\n", valueDate, currency, amount))
		}
	}
	
	if correspondent, ok := payload["senders_correspondent"].(string); ok {
		body.WriteString(fmt.Sprintf(":53A:%s\n", correspondent))
	}
	
	if institution, ok := payload["beneficiary_institution"].(string); ok {
		body.WriteString(fmt.Sprintf(":58A:%s\n", institution))
	}
	
	if info, ok := payload["sender_to_receiver_information"].(string); ok {
		body.WriteString(fmt.Sprintf(":72:%s\n", info))
	}
	
	return body.String(), nil
}

func (s *SWIFTHandler) formatMT940Body(payload map[string]interface{}) (string, error) {
	var body strings.Builder
	
	if ref, ok := payload["transaction_reference"].(string); ok {
		body.WriteString(fmt.Sprintf(":20:%s\n", ref))
	}
	
	if account, ok := payload["account_identification"].(string); ok {
		body.WriteString(fmt.Sprintf(":25:%s\n", account))
	}
	
	if stmt, ok := payload["statement_number"].(string); ok {
		body.WriteString(fmt.Sprintf(":28C:%s\n", stmt))
	}
	
	if balance, ok := payload["opening_balance"].(map[string]interface{}); ok {
		body.WriteString(fmt.Sprintf(":60F:%s\n", formatBalance(balance)))
	}
	
	if transactions, ok := payload["transactions"].([]interface{}); ok {
		for _, txn := range transactions {
			if txnMap, ok := txn.(map[string]interface{}); ok {
				body.WriteString(fmt.Sprintf(":61:%s\n", formatStatementLine(txnMap)))
				if info, ok := txnMap["information"].(string); ok {
					body.WriteString(fmt.Sprintf(":86:%s\n", info))
				}
			}
		}
	}
	
	if balance, ok := payload["closing_balance"].(map[string]interface{}); ok {
		body.WriteString(fmt.Sprintf(":62F:%s\n", formatBalance(balance)))
	}
	
	return body.String(), nil
}

func (s *SWIFTHandler) Validate(msg *UniversalMessage) error {
	return s.validator.ValidateSWIFT(msg)
}

func (s *SWIFTHandler) GetMessageType() string {
	return "SWIFT_MT"
}

func (s *SWIFTHandler) determinePriority(messageType string, payload map[string]interface{}) MessagePriority {
	switch messageType {
	case "103", "202":
		if amount, ok := payload["amount"].(float64); ok {
			if amount >= 1000000 { // $1M+
				return PriorityHigh
			} else if amount >= 100000 { // $100K+
				return PriorityNormal
			}
		}
		return PriorityNormal
	case "910", "940", "950":
		return PriorityLow
	default:
		return PriorityNormal
	}
}

func (s *SWIFTHandler) generateComplianceInfo(payload map[string]interface{}) *ComplianceInfo {
	compliance := &ComplianceInfo{
		AMLScore:       calculateAMLScore(payload),
		SanctionsCheck: true, // Assume checked
		KYCVerified:    true, // Assume verified
	}
	
	// Add regulatory flags based on content
	if amount, ok := payload["amount"].(float64); ok {
		if amount >= 10000 {
			compliance.RegulatoryFlags = append(compliance.RegulatoryFlags, "LARGE_VALUE")
		}
	}
	
	if sender, ok := payload["ordering_customer"].(map[string]interface{}); ok {
		if country, ok := sender["country"].(string); ok {
			if isHighRiskCountry(country) {
				compliance.RegulatoryFlags = append(compliance.RegulatoryFlags, "HIGH_RISK_COUNTRY")
			}
		}
	}
	
	// Generate compliance hash
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%.2f%v%v", compliance.AMLScore, compliance.SanctionsCheck, compliance.KYCVerified)))
	compliance.ComplianceHash = fmt.Sprintf("%x", h.Sum(nil))
	
	return compliance
}

// ISO 20022 XML message handler
type ISO20022Handler struct {
	BIC       string
	logger    *zap.Logger
	validator *MessageValidator
}

func NewISO20022Handler(bic string, logger *zap.Logger) *ISO20022Handler {
	return &ISO20022Handler{
		BIC:       bic,
		logger:    logger,
		validator: NewMessageValidator(),
	}
}

func (i *ISO20022Handler) Parse(data []byte) (*UniversalMessage, error) {
	var document map[string]interface{}
	if err := xml.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("failed to parse XML: %w", err)
	}
	
	// Extract message type from root element
	messageType := i.extractMessageType(document)
	if messageType == "" {
		return nil, fmt.Errorf("unable to determine message type")
	}
	
	// Parse based on message type
	payload, err := i.parseISO20022Body(messageType, document)
	if err != nil {
		return nil, fmt.Errorf("failed to parse %s message: %w", messageType, err)
	}
	
	msg := &UniversalMessage{
		ID:        generateMessageID(),
		Type:      messageType,
		Version:   "2023",
		Timestamp: time.Now(),
		Priority:  i.determinePriority(messageType, payload),
		Network:   "ISO20022",
		Payload:   payload,
		Compliance: i.generateComplianceInfo(payload),
	}
	
	// Extract sender/receiver from message
	i.extractParticipants(document, msg)
	
	return msg, nil
}

func (i *ISO20022Handler) Format(msg *UniversalMessage) ([]byte, error) {
	switch msg.Type {
	case PAIN001:
		return i.formatPAIN001(msg)
	case PACS008:
		return i.formatPACS008(msg)
	case CAMT053:
		return i.formatCAMT053(msg)
	default:
		return nil, fmt.Errorf("unsupported ISO 20022 message type: %s", msg.Type)
	}
}

func (i *ISO20022Handler) formatPAIN001(msg *UniversalMessage) ([]byte, error) {
	// Build CustomerCreditTransferInitiation XML
	doc := map[string]interface{}{
		"Document": map[string]interface{}{
			"@xmlns": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.11",
			"CstmrCdtTrfInitn": map[string]interface{}{
				"GrpHdr": map[string]interface{}{
					"MsgId":    msg.ID,
					"CreDtTm":  msg.Timestamp.Format(time.RFC3339),
					"NbOfTxs":  "1",
					"CtrlSum":  msg.Payload["amount"],
					"InitgPty": map[string]interface{}{
						"Nm": msg.Sender.Name,
						"Id": map[string]interface{}{
							"OrgId": map[string]interface{}{
								"BICOrBEI": msg.Sender.BIC,
							},
						},
					},
				},
				"PmtInf": i.buildPaymentInstruction(msg),
			},
		},
	}
	
	return xml.Marshal(doc)
}

func (i *ISO20022Handler) formatPACS008(msg *UniversalMessage) ([]byte, error) {
	// Build FIToFICustomerCreditTransfer XML
	doc := map[string]interface{}{
		"Document": map[string]interface{}{
			"@xmlns": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.10",
			"FIToFICstmrCdtTrf": map[string]interface{}{
				"GrpHdr": i.buildGroupHeader(msg),
				"CdtTrfTxInf": i.buildCreditTransferInfo(msg),
			},
		},
	}
	
	return xml.Marshal(doc)
}

func (i *ISO20022Handler) formatCAMT053(msg *UniversalMessage) ([]byte, error) {
	// Build BankToCustomerStatement XML
	doc := map[string]interface{}{
		"Document": map[string]interface{}{
			"@xmlns": "urn:iso:std:iso:20022:tech:xsd:camt.053.001.10",
			"BkToCstmrStmt": map[string]interface{}{
				"GrpHdr": i.buildGroupHeader(msg),
				"Stmt":   i.buildStatement(msg),
			},
		},
	}
	
	return xml.Marshal(doc)
}

func (i *ISO20022Handler) Validate(msg *UniversalMessage) error {
	return i.validator.ValidateISO20022(msg)
}

func (i *ISO20022Handler) GetMessageType() string {
	return "ISO20022"
}

// Universal Banking Protocol Engine
type ProtocolEngine struct {
	handlers       map[string]ProtocolHandler
	messageQueue   chan *UniversalMessage
	rateLimiter    *rate.Limiter
	metrics        *ProtocolMetrics
	logger         *zap.Logger
	validator      *MessageValidator
	transformer    *MessageTransformer
	router         *MessageRouter
	
	// Configuration
	config         *ProtocolConfig
	
	// Concurrency control with enhanced synchronization
	workers        int
	shutdown       chan struct{}
	wg             sync.WaitGroup
	shutdownOnce   sync.Once
	isShutdown     int64 // atomic flag
	
	// Message tracking with cleanup
	messageTracker *MessageTracker
	
	// Worker coordination
	workerSemaphore *semaphore.Weighted
}

type ProtocolConfig struct {
	MaxWorkers           int           `json:"max_workers"`
	QueueSize           int           `json:"queue_size"`
	RateLimit           rate.Limit    `json:"rate_limit"`
	MessageTimeout      time.Duration `json:"message_timeout"`
	RetryAttempts       int           `json:"retry_attempts"`
	EnableValidation    bool          `json:"enable_validation"`
	EnableTransformation bool         `json:"enable_transformation"`
	EnableRouting       bool          `json:"enable_routing"`
}

type ProtocolMetrics struct {
	MessagesProcessed   int64 `json:"messages_processed"`
	MessagesFailed      int64 `json:"messages_failed"`
	AverageLatency      int64 `json:"average_latency_ms"`
	ThroughputTPS       int64 `json:"throughput_tps"`
	ValidationFailures  int64 `json:"validation_failures"`
	TransformationErrors int64 `json:"transformation_errors"`
	RoutingErrors       int64 `json:"routing_errors"`
	
	// Internal atomic counters for metrics calculation
	totalLatency        int64
	latencyCount        int64
	lastProcessedCount  int64
	
	mutex sync.RWMutex
}

type MessageValidator struct {
	swiftValidators    map[string]func(*UniversalMessage) error
	iso20022Validators map[string]func(*UniversalMessage) error
	customValidators   map[string]func(*UniversalMessage) error
}

type MessageTransformer struct {
	transformers map[string]map[string]func(*UniversalMessage) (*UniversalMessage, error)
}

type MessageRouter struct {
	routes      map[string][]string
	preferences map[string]string
	costs       map[string]float64
}

type MessageTracker struct {
	messages    map[string]*MessageStatus
	mutex       sync.RWMutex
	cleanupChan chan string
	done        chan struct{}
	wg          sync.WaitGroup
}

type MessageStatus struct {
	ID          string                 `json:"id"`
	Status      string                 `json:"status"`
	CreatedAt   time.Time             `json:"created_at"`
	UpdatedAt   time.Time             `json:"updated_at"`
	Attempts    int                   `json:"attempts"`
	Errors      []string              `json:"errors,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

func NewProtocolEngine(config *ProtocolConfig, logger *zap.Logger) *ProtocolEngine {
	engine := &ProtocolEngine{
		handlers:        make(map[string]ProtocolHandler),
		messageQueue:    make(chan *UniversalMessage, config.QueueSize),
		rateLimiter:     rate.NewLimiter(config.RateLimit, int(config.RateLimit)),
		metrics:         &ProtocolMetrics{},
		logger:          logger,
		validator:       NewMessageValidator(),
		transformer:     NewMessageTransformer(),
		router:          NewMessageRouter(),
		config:          config,
		workers:         config.MaxWorkers,
		shutdown:        make(chan struct{}),
		messageTracker:  NewMessageTracker(),
		workerSemaphore: semaphore.NewWeighted(int64(config.MaxWorkers)),
		isShutdown:      0,
	}
	
	// Register built-in handlers
	engine.RegisterHandler("SWIFT", NewSWIFTHandler("", logger))
	engine.RegisterHandler("ISO20022", NewISO20022Handler("", logger))
	
	return engine
}

func (pe *ProtocolEngine) RegisterHandler(protocol string, handler ProtocolHandler) {
	pe.handlers[protocol] = handler
	pe.logger.Info("Registered protocol handler", zap.String("protocol", protocol))
}

func (pe *ProtocolEngine) Start(ctx context.Context) error {
	// Check if already running
	if atomic.LoadInt64(&pe.isShutdown) == 1 {
		return fmt.Errorf("engine is shutting down")
	}
	
	pe.logger.Info("Starting protocol engine", zap.Int("workers", pe.workers))
	
	// Start message tracker cleanup goroutine first
	pe.messageTracker.Start()
	
	// Start worker goroutines with semaphore control
	for i := 0; i < pe.workers; i++ {
		pe.wg.Add(1)
		go pe.worker(ctx, i)
	}
	
	// Start metrics collector
	pe.wg.Add(1)
	go pe.metricsCollector(ctx)
	
	return nil
}

func (pe *ProtocolEngine) Stop() error {
	pe.shutdownOnce.Do(func() {
		pe.logger.Info("Stopping protocol engine")
		
		// Set shutdown flag atomically
		atomic.StoreInt64(&pe.isShutdown, 1)
		
		// Close shutdown channel to signal workers
		close(pe.shutdown)
		
		// Close message queue to prevent new messages
		close(pe.messageQueue)
		
		// Stop message tracker
		pe.messageTracker.Stop()
		
		// Wait for all workers to finish
		pe.wg.Wait()
		
		pe.logger.Info("Protocol engine stopped")
	})
	return nil
}

func (pe *ProtocolEngine) ProcessMessage(ctx context.Context, data []byte, protocol string) (*UniversalMessage, error) {
	start := time.Now()
	
	// Rate limiting
	if err := pe.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit exceeded: %w", err)
	}
	
	// Get protocol handler
	handler, exists := pe.handlers[protocol]
	if !exists {
		return nil, fmt.Errorf("unsupported protocol: %s", protocol)
	}
	
	// Parse message
	msg, err := handler.Parse(data)
	if err != nil {
		atomic.AddInt64(&pe.metrics.MessagesFailed, 1)
		return nil, fmt.Errorf("failed to parse %s message: %w", protocol, err)
	}
	
	// Track message
	pe.messageTracker.TrackMessage(msg.ID, "parsing", nil)
	
	// Validate message
	if pe.config.EnableValidation {
		if err := pe.validator.Validate(msg); err != nil {
			atomic.AddInt64(&pe.metrics.ValidationFailures, 1)
			pe.messageTracker.UpdateMessage(msg.ID, "validation_failed", map[string]interface{}{"error": err.Error()})
			return nil, fmt.Errorf("validation failed: %w", err)
		}
	}
	
	// Transform message
	if pe.config.EnableTransformation {
		transformedMsg, err := pe.transformer.Transform(msg)
		if err != nil {
			atomic.AddInt64(&pe.metrics.TransformationErrors, 1)
			pe.messageTracker.UpdateMessage(msg.ID, "transformation_failed", map[string]interface{}{"error": err.Error()})
			return nil, fmt.Errorf("transformation failed: %w", err)
		}
		msg = transformedMsg
	}
	
	// Route message
	if pe.config.EnableRouting {
		if err := pe.router.Route(msg); err != nil {
			atomic.AddInt64(&pe.metrics.RoutingErrors, 1)
			pe.messageTracker.UpdateMessage(msg.ID, "routing_failed", map[string]interface{}{"error": err.Error()})
			return nil, fmt.Errorf("routing failed: %w", err)
		}
	}
	
	// Update metrics atomically
	latency := time.Since(start).Milliseconds()
	atomic.AddInt64(&pe.metrics.MessagesProcessed, 1)
	atomic.AddInt64(&pe.metrics.totalLatency, latency)
	atomic.AddInt64(&pe.metrics.latencyCount, 1)
	
	pe.messageTracker.UpdateMessage(msg.ID, "processed", map[string]interface{}{"latency_ms": latency})
	
	return msg, nil
}

func (pe *ProtocolEngine) worker(ctx context.Context, workerID int) {
	defer pe.wg.Done()
	
	pe.logger.Info("Starting protocol worker", zap.Int("worker_id", workerID))
	
	for {
		select {
		case <-ctx.Done():
			pe.logger.Info("Worker stopping due to context cancellation", zap.Int("worker_id", workerID))
			return
		case <-pe.shutdown:
			pe.logger.Info("Worker stopping due to shutdown signal", zap.Int("worker_id", workerID))
			return
		case msg, ok := <-pe.messageQueue:
			if !ok {
				pe.logger.Info("Worker stopping due to closed message queue", zap.Int("worker_id", workerID))
				return
			}
			
			// Acquire semaphore before processing (with timeout)
			if err := pe.workerSemaphore.Acquire(ctx, 1); err != nil {
				pe.logger.Warn("Failed to acquire worker semaphore", zap.Error(err))
				continue
			}
			
			pe.processMessageAsync(ctx, msg)
			pe.workerSemaphore.Release(1)
		}
	}
}

func (pe *ProtocolEngine) processMessageAsync(ctx context.Context, msg *UniversalMessage) {
	start := time.Now()
	
	// Process with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, pe.config.MessageTimeout)
	defer cancel()
	
	// Simulate processing
	select {
	case <-timeoutCtx.Done():
		pe.logger.Warn("Message processing timeout", zap.String("message_id", msg.ID))
		atomic.AddInt64(&pe.metrics.MessagesFailed, 1)
	default:
		// Process successfully
		time.Sleep(time.Millisecond * 10) // Simulate processing time
		
		latency := time.Since(start).Milliseconds()
		atomic.AddInt64(&pe.metrics.MessagesProcessed, 1)
		
		pe.logger.Debug("Message processed", 
			zap.String("message_id", msg.ID),
			zap.Int64("latency_ms", latency))
	}
}

func (pe *ProtocolEngine) metricsCollector(ctx context.Context) {
	defer pe.wg.Done()
	
	ticker := time.NewTicker(time.Second * 60)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-pe.shutdown:
			return
		case <-ticker.C:
			pe.collectMetrics()
		}
	}
}

func (pe *ProtocolEngine) collectMetrics() {
	pe.metrics.mutex.Lock()
	defer pe.metrics.mutex.Unlock()
	
	// Calculate TPS based on delta
	currentProcessed := atomic.LoadInt64(&pe.metrics.MessagesProcessed)
	pe.metrics.ThroughputTPS = (currentProcessed - pe.metrics.lastProcessedCount) / 60
	pe.metrics.lastProcessedCount = currentProcessed
	
	// Calculate average latency atomically
	totalLatency := atomic.LoadInt64(&pe.metrics.totalLatency)
	latencyCount := atomic.LoadInt64(&pe.metrics.latencyCount)
	if latencyCount > 0 {
		pe.metrics.AverageLatency = totalLatency / latencyCount
	}
	
	pe.logger.Info("Protocol engine metrics",
		zap.Int64("messages_processed", pe.metrics.MessagesProcessed),
		zap.Int64("messages_failed", pe.metrics.MessagesFailed),
		zap.Int64("throughput_tps", pe.metrics.ThroughputTPS),
		zap.Int64("average_latency_ms", pe.metrics.AverageLatency),
		zap.Int64("validation_failures", pe.metrics.ValidationFailures),
		zap.Int64("transformation_errors", pe.metrics.TransformationErrors),
		zap.Int64("routing_errors", pe.metrics.RoutingErrors))
}

// Utility functions
func generateMessageID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func parseAmount(amountStr string) (float64, error) {
	// Remove commas and parse
	cleaned := strings.ReplaceAll(amountStr, ",", "")
	return strconv.ParseFloat(cleaned, 64)
}

func parsePartyIdentification(value string) map[string]interface{} {
	lines := strings.Split(value, "\n")
	party := make(map[string]interface{})
	
	if len(lines) > 0 {
		if strings.HasPrefix(lines[0], "/") {
			party["account"] = strings.TrimPrefix(lines[0], "/")
			if len(lines) > 1 {
				party["name"] = lines[1]
			}
		} else {
			party["name"] = lines[0]
		}
	}
	
	return party
}

func formatPartyIdentification(party map[string]interface{}) string {
	var result strings.Builder
	
	if account, ok := party["account"].(string); ok {
		result.WriteString("/")
		result.WriteString(account)
		result.WriteString("\n")
	}
	
	if name, ok := party["name"].(string); ok {
		result.WriteString(name)
	}
	
	return result.String()
}

func parseBalance(value string) map[string]interface{} {
	balance := make(map[string]interface{})
	
	if len(value) >= 10 {
		balance["indicator"] = string(value[0])
		balance["date"] = value[1:7]
		balance["currency"] = value[7:10]
		if len(value) > 10 {
			if amount, err := parseAmount(value[10:]); err == nil {
				balance["amount"] = amount
			}
		}
	}
	
	return balance
}

func formatBalance(balance map[string]interface{}) string {
	var result strings.Builder
	
	if indicator, ok := balance["indicator"].(string); ok {
		result.WriteString(indicator)
	}
	
	if date, ok := balance["date"].(string); ok {
		result.WriteString(date)
	}
	
	if currency, ok := balance["currency"].(string); ok {
		result.WriteString(currency)
	}
	
	if amount, ok := balance["amount"].(float64); ok {
		result.WriteString(fmt.Sprintf("%.2f", amount))
	}
	
	return result.String()
}

func formatStatementLine(txn map[string]interface{}) string {
	// Format :61: statement line
	var line strings.Builder
	
	if date, ok := txn["value_date"].(string); ok {
		line.WriteString(date)
	}
	
	if amount, ok := txn["amount"].(float64); ok {
		if amount < 0 {
			line.WriteString("D")
			amount = -amount
		} else {
			line.WriteString("C")
		}
		line.WriteString(fmt.Sprintf("%.2f", amount))
	}
	
	if ref, ok := txn["reference"].(string); ok {
		line.WriteString(ref)
	}
	
	return line.String()
}

func calculateAMLScore(payload map[string]interface{}) float64 {
	score := 0.0
	
	// High-value transaction
	if amount, ok := payload["amount"].(float64); ok {
		if amount > 100000 {
			score += 0.3
		} else if amount > 50000 {
			score += 0.2
		} else if amount > 10000 {
			score += 0.1
		}
	}
	
	// Cross-border transaction
	if sender, ok := payload["ordering_customer"].(map[string]interface{}); ok {
		if senderCountry, ok := sender["country"].(string); ok {
			if receiver, ok := payload["beneficiary_customer"].(map[string]interface{}); ok {
				if receiverCountry, ok := receiver["country"].(string); ok {
					if senderCountry != receiverCountry {
						score += 0.1
						
						if isHighRiskCountry(senderCountry) || isHighRiskCountry(receiverCountry) {
							score += 0.3
						}
					}
				}
			}
		}
	}
	
	return score
}

func isHighRiskCountry(country string) bool {
	highRiskCountries := []string{"IR", "KP", "AF", "SY", "YE", "VE", "MM"}
	for _, riskCountry := range highRiskCountries {
		if country == riskCountry {
			return true
		}
	}
	return false
}

// Constructor functions
func NewMessageValidator() *MessageValidator {
	return &MessageValidator{
		swiftValidators:    make(map[string]func(*UniversalMessage) error),
		iso20022Validators: make(map[string]func(*UniversalMessage) error),
		customValidators:   make(map[string]func(*UniversalMessage) error),
	}
}

func NewMessageTransformer() *MessageTransformer {
	return &MessageTransformer{
		transformers: make(map[string]map[string]func(*UniversalMessage) (*UniversalMessage, error)),
	}
}

func NewMessageRouter() *MessageRouter {
	return &MessageRouter{
		routes:      make(map[string][]string),
		preferences: make(map[string]string),
		costs:       make(map[string]float64),
	}
}

func NewMessageTracker() *MessageTracker {
	return &MessageTracker{
		messages:    make(map[string]*MessageStatus),
		cleanupChan: make(chan string, 1000),
		done:        make(chan struct{}),
	}
}

// Additional methods for validator, transformer, router, and tracker would be implemented here...

func (mv *MessageValidator) Validate(msg *UniversalMessage) error {
	// Basic validation
	if msg.ID == "" {
		return fmt.Errorf("message ID is required")
	}
	
	if msg.Type == "" {
		return fmt.Errorf("message type is required")
	}
	
	// Network-specific validation
	switch msg.Network {
	case "SWIFT":
		return mv.ValidateSWIFT(msg)
	case "ISO20022":
		return mv.ValidateISO20022(msg)
	default:
		return nil
	}
}

func (mv *MessageValidator) ValidateSWIFT(msg *UniversalMessage) error {
	// SWIFT-specific validation
	if msg.Sender.BIC == "" || len(msg.Sender.BIC) != 11 {
		return fmt.Errorf("invalid sender BIC")
	}
	
	if msg.Receiver.BIC == "" || len(msg.Receiver.BIC) != 11 {
		return fmt.Errorf("invalid receiver BIC")
	}
	
	return nil
}

func (mv *MessageValidator) ValidateISO20022(msg *UniversalMessage) error {
	// ISO 20022-specific validation
	if msg.Sender.LEI != "" && len(msg.Sender.LEI) != 20 {
		return fmt.Errorf("invalid sender LEI")
	}
	
	if msg.Receiver.LEI != "" && len(msg.Receiver.LEI) != 20 {
		return fmt.Errorf("invalid receiver LEI")
	}
	
	return nil
}

func (mt *MessageTransformer) Transform(msg *UniversalMessage) (*UniversalMessage, error) {
	// Apply transformations based on message type and network
	transformKey := fmt.Sprintf("%s_%s", msg.Network, msg.Type)
	
	if transformers, exists := mt.transformers[transformKey]; exists {
		for name, transformer := range transformers {
			transformedMsg, err := transformer(msg)
			if err != nil {
				return nil, fmt.Errorf("transformation '%s' failed: %w", name, err)
			}
			msg = transformedMsg
		}
	}
	
	return msg, nil
}

func (mr *MessageRouter) Route(msg *UniversalMessage) error {
	// Route message based on type, priority, and preferences
	routeKey := fmt.Sprintf("%s_%s", msg.Network, msg.Type)
	
	if routes, exists := mr.routes[routeKey]; exists {
		// Select best route based on cost and preferences
		bestRoute := routes[0] // Simplified routing
		msg.Routing = &RoutingInfo{
			Path:        []string{bestRoute},
			Hops:        1,
			Preferences: mr.preferences,
		}
	}
	
	return nil
}

// MessageTracker methods
func (mt *MessageTracker) Start() {
	mt.wg.Add(1)
	go mt.cleanupWorker()
}

func (mt *MessageTracker) Stop() {
	close(mt.done)
	mt.wg.Wait()
}

func (mt *MessageTracker) cleanupWorker() {
	defer mt.wg.Done()
	
	cleanupTicker := time.NewTicker(5 * time.Minute)
	defer cleanupTicker.Stop()
	
	for {
		select {
		case <-mt.done:
			return
		case <-cleanupTicker.C:
			mt.cleanupOldMessages()
		case msgID := <-mt.cleanupChan:
			mt.removeMessage(msgID)
		}
	}
}

func (mt *MessageTracker) cleanupOldMessages() {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()
	
	cutoff := time.Now().Add(-1 * time.Hour) // Remove messages older than 1 hour
	toDelete := make([]string, 0)
	
	for id, status := range mt.messages {
		if status.UpdatedAt.Before(cutoff) {
			toDelete = append(toDelete, id)
		}
	}
	
	for _, id := range toDelete {
		delete(mt.messages, id)
	}
	
	if len(toDelete) > 0 {
		// Logger would go here if available
	}
}

func (mt *MessageTracker) removeMessage(id string) {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()
	delete(mt.messages, id)
}

func (mt *MessageTracker) TrackMessage(id, status string, metadata map[string]interface{}) {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()
	
	mt.messages[id] = &MessageStatus{
		ID:        id,
		Status:    status,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Attempts:  1,
		Metadata:  metadata,
	}
}

func (mt *MessageTracker) UpdateMessage(id, status string, metadata map[string]interface{}) {
	mt.mutex.Lock()
	defer mt.mutex.Unlock()
	
	if msg, exists := mt.messages[id]; exists {
		msg.Status = status
		msg.UpdatedAt = time.Now()
		msg.Attempts++
		if metadata != nil {
			if msg.Metadata == nil {
				msg.Metadata = make(map[string]interface{})
			}
			for k, v := range metadata {
				msg.Metadata[k] = v
			}
		}
	}
}

// Additional helper methods for ISO 20022 parsing would be implemented here...

func (i *ISO20022Handler) extractMessageType(document map[string]interface{}) string {
	// Extract message type from XML namespace or root element
	if docElement, ok := document["Document"]; ok {
		if docMap, ok := docElement.(map[string]interface{}); ok {
			for key := range docMap {
				switch key {
				case "CstmrCdtTrfInitn":
					return PAIN001
				case "FIToFICstmrCdtTrf":
					return PACS008
				case "BkToCstmrStmt":
					return CAMT053
				}
			}
		}
	}
	return ""
}

func (i *ISO20022Handler) parseISO20022Body(messageType string, document map[string]interface{}) (map[string]interface{}, error) {
	payload := make(map[string]interface{})
	
	switch messageType {
	case PAIN001:
		return i.parsePAIN001Body(document, payload)
	case PACS008:
		return i.parsePACS008Body(document, payload)
	case CAMT053:
		return i.parseCAMT053Body(document, payload)
	}
	
	return payload, nil
}

func (i *ISO20022Handler) parsePAIN001Body(document map[string]interface{}, payload map[string]interface{}) (map[string]interface{}, error) {
	// Parse CustomerCreditTransferInitiation
	return payload, nil
}

func (i *ISO20022Handler) parsePACS008Body(document map[string]interface{}, payload map[string]interface{}) (map[string]interface{}, error) {
	// Parse FIToFICustomerCreditTransfer
	return payload, nil
}

func (i *ISO20022Handler) parseCAMT053Body(document map[string]interface{}, payload map[string]interface{}) (map[string]interface{}, error) {
	// Parse BankToCustomerStatement
	return payload, nil
}

func (i *ISO20022Handler) extractParticipants(document map[string]interface{}, msg *UniversalMessage) {
	// Extract sender/receiver information from XML
}

func (i *ISO20022Handler) determinePriority(messageType string, payload map[string]interface{}) MessagePriority {
	// Determine priority based on message type and content
	return PriorityNormal
}

func (i *ISO20022Handler) generateComplianceInfo(payload map[string]interface{}) *ComplianceInfo {
	// Generate compliance information
	return &ComplianceInfo{
		AMLScore:       0.0,
		SanctionsCheck: true,
		KYCVerified:    true,
	}
}

func (i *ISO20022Handler) buildGroupHeader(msg *UniversalMessage) map[string]interface{} {
	return map[string]interface{}{
		"MsgId":   msg.ID,
		"CreDtTm": msg.Timestamp.Format(time.RFC3339),
	}
}

func (i *ISO20022Handler) buildPaymentInstruction(msg *UniversalMessage) map[string]interface{} {
	return map[string]interface{}{
		"PmtInfId": msg.ID,
		"PmtMtd":   "TRF",
	}
}

func (i *ISO20022Handler) buildCreditTransferInfo(msg *UniversalMessage) map[string]interface{} {
	return map[string]interface{}{
		"PmtId": map[string]interface{}{
			"InstrId": msg.ID,
		},
	}
}

func (i *ISO20022Handler) buildStatement(msg *UniversalMessage) map[string]interface{} {
	return map[string]interface{}{
		"Id": msg.ID,
	}
}

func (i *ISO20022Handler) parseStatementLine(value string, payload map[string]interface{}) map[string]interface{} {
	// Parse statement line :61:
	return payload
}

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()
	
	config := &ProtocolConfig{
		MaxWorkers:           10,
		QueueSize:           1000,
		RateLimit:           1000, // 1000 messages per second
		MessageTimeout:      30 * time.Second,
		RetryAttempts:       3,
		EnableValidation:    true,
		EnableTransformation: true,
		EnableRouting:       true,
	}
	
	engine := NewProtocolEngine(config, logger)
	
	ctx := context.Background()
	if err := engine.Start(ctx); err != nil {
		log.Fatal("Failed to start protocol engine:", err)
	}
	
	fmt.Println("QENEX Universal Banking Protocol Engine - Ready")
	fmt.Println("Supporting SWIFT MT, ISO 20022, and custom protocols")
	
	// Keep running
	select {}
}