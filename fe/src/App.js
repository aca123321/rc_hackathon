import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [sessionMemory, setSessionMemory] = useState(null);
  const [isMemoryExpanded, setIsMemoryExpanded] = useState(false);
  const [topK, setTopK] = useState(10);
  const [historyLength, setHistoryLength] = useState(3);
  const [maxChunksPerSource, setMaxChunksPerSource] = useState(5);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.0);
  const [maxCompletionTokens, setMaxCompletionTokens] = useState(1500);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [cacheStats, setCacheStats] = useState(null);
  const messagesEndRef = useRef(null);
  const dropdownRef = useRef(null);
  const settingsRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (response.ok) {
          const data = await response.json();
          setModels(data.models);
          setSelectedModel(data.current_model);
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      } finally {
        setModelsLoading(false);
      }
    };
    fetchModels();
  }, []);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsModelDropdownOpen(false);
      }
      if (settingsRef.current && !settingsRef.current.contains(event.target)) {
        setIsSettingsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fetch cache stats when settings dropdown opens
  useEffect(() => {
    if (isSettingsOpen) {
      fetch(`${API_BASE_URL}/cache-stats`)
        .then((res) => res.json())
        .then((data) => setCacheStats(data))
        .catch(() => setCacheStats(null));
    }
  }, [isSettingsOpen]);

  const toggleSources = (messageId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleModelChange = async (modelId) => {
    setSelectedModel(modelId);
    setIsModelDropdownOpen(false);

    // Update model on server
    try {
      await fetch(`${API_BASE_URL}/model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelId }),
      });
    } catch (error) {
      console.error("Error setting model:", error);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: inputValue.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage.content,
          model: selectedModel,
          top_k: topK,
          history_length: historyLength,
          max_chunks_per_source: maxChunksPerSource,
          similarity_threshold: similarityThreshold,
          max_completion_tokens: maxCompletionTokens,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content: data.answer,
        sources: data.source_documents || [],
        sections: data.relevant_sections || [],
        model: selectedModel,
        cost: data.cost || null,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Update session memory from response
      if (data.session_memory) {
        setSessionMemory(data.session_memory);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content:
          "Sorry, I encountered an error connecting to the server. Please make sure the backend is running.",
        isError: true,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = async () => {
    try {
      await fetch(`${API_BASE_URL}/clear-history`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Error clearing history on server:", error);
    }
    setMessages([]);
    setExpandedSources({});
    setSessionMemory(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatSources = (sources, sections) => {
    const uniqueSources = [...new Set(sources)];
    const uniqueSections = [...new Set(sections)];
    return { uniqueSources, uniqueSections };
  };

  const getSelectedModelInfo = () => {
    return models.find((m) => m.id === selectedModel);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">
              NovaTech Assistant
            </h1>
            <p className="text-xs text-gray-500">Powered by RAG</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Model Selector */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
              disabled={modelsLoading}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <svg
                className="w-4 h-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
              <span className="text-gray-700 font-medium max-w-[120px] truncate">
                {modelsLoading
                  ? "Loading..."
                  : getSelectedModelInfo()?.name || selectedModel}
              </span>
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${isModelDropdownOpen ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>

            {/* Dropdown Menu */}
            {isModelDropdownOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden animate-fadeIn">
                <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    Select Model
                  </p>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {models.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => handleModelChange(model.id)}
                      className={`w-full px-3 py-3 text-left hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-b-0 ${
                        selectedModel === model.id ? "bg-blue-50" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-gray-900">
                              {model.name}
                            </span>
                            {selectedModel === model.id && (
                              <span className="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                                Active
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5 truncate">
                            {model.description}
                          </p>
                          <p className="text-xs text-gray-400 mt-1">
                            {model.recommended_for}
                          </p>
                        </div>
                        <div className="ml-3 text-right flex-shrink-0">
                          <span className="text-xs text-gray-400">
                            {(model.context_window / 1000).toFixed(0)}K ctx
                          </span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Settings Button */}
          <div className="relative" ref={settingsRef}>
            <button
              onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors flex items-center gap-2"
              title="Retrieval Settings"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                />
              </svg>
              <span className="hidden sm:inline">Settings</span>
            </button>

            {/* Settings Dropdown */}
            {isSettingsOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-white border border-gray-200 rounded-xl shadow-lg z-50 overflow-hidden animate-fadeIn max-h-[80vh] overflow-y-auto">
                <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 sticky top-0">
                  <p className="text-sm font-semibold text-gray-700">
                    Retrieval & Generation Settings
                  </p>
                </div>
                <div className="p-4 space-y-4">
                  {/* Top K Setting */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Chunks to Retrieve
                      </label>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        {topK}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="3"
                      max="20"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>3</span>
                      <span>20</span>
                    </div>
                  </div>

                  {/* Max Chunks Per Source Setting */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Max Chunks Per Source
                      </label>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        {maxChunksPerSource}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={maxChunksPerSource}
                      onChange={(e) =>
                        setMaxChunksPerSource(parseInt(e.target.value))
                      }
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>1 (diverse)</span>
                      <span>5 (focused)</span>
                    </div>
                  </div>

                  {/* Similarity Threshold Setting */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Similarity Threshold
                      </label>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        {similarityThreshold.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="0.8"
                      step="0.05"
                      value={similarityThreshold}
                      onChange={(e) =>
                        setSimilarityThreshold(parseFloat(e.target.value))
                      }
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>0 (all)</span>
                      <span>0.8 (strict)</span>
                    </div>
                  </div>

                  {/* History Length Setting */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Message History
                      </label>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        {historyLength} pairs
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      value={historyLength}
                      onChange={(e) =>
                        setHistoryLength(parseInt(e.target.value))
                      }
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>0</span>
                      <span>10</span>
                    </div>
                  </div>

                  {/* Max Completion Tokens Setting */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        Max Response Tokens
                      </label>
                      <span className="text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        {maxCompletionTokens}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="500"
                      max="4000"
                      step="100"
                      value={maxCompletionTokens}
                      onChange={(e) =>
                        setMaxCompletionTokens(parseInt(e.target.value))
                      }
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>500</span>
                      <span>4000</span>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-gray-100 space-y-2">
                    <p className="text-xs text-gray-500">
                      Higher values = more tokens = higher cost.
                    </p>
                    {cacheStats && (
                      <div className="text-xs text-gray-500 bg-gray-50 rounded p-2">
                        <span className="font-medium">Cache:</span>{" "}
                        {cacheStats.size}/{cacheStats.max_size} entries,{" "}
                        {cacheStats.hit_rate} hit rate
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Clear Chat Button */}
          <button
            onClick={clearConversation}
            className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Clear Chat
          </button>
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-8 h-8 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-medium text-gray-900 mb-2">
                Welcome to NovaTech Assistant
              </h2>
              <p className="text-gray-500 max-w-md mx-auto">
                Ask me anything about NovaTech products, documentation, or
                policies. I'll provide answers with source citations.
              </p>
              {selectedModel && (
                <p className="text-xs text-gray-400 mt-4">
                  Using {getSelectedModelInfo()?.name || selectedModel}
                </p>
              )}
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] ${
                  message.role === "user"
                    ? "bg-blue-600 text-white rounded-2xl rounded-br-md"
                    : message.isError
                      ? "bg-red-50 text-red-800 border border-red-200 rounded-2xl rounded-bl-md"
                      : "bg-white text-gray-900 border border-gray-200 rounded-2xl rounded-bl-md shadow-sm"
                } px-4 py-3`}
              >
                {message.role === "user" ? (
                  <p className="whitespace-pre-wrap">{message.content}</p>
                ) : (
                  <div className="markdown-content prose prose-sm max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                )}

                {/* Model and cost indicator for assistant messages */}
                {message.role === "assistant" &&
                  !message.isError &&
                  (message.model || message.cost) && (
                    <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
                      {message.model && (
                        <div className="flex items-center gap-1">
                          <svg
                            className="w-3 h-3"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                            />
                          </svg>
                          <span>
                            {models.find((m) => m.id === message.model)?.name ||
                              message.model}
                          </span>
                        </div>
                      )}
                      {message.cost && (
                        <div
                          className="flex items-center gap-1"
                          title={`${message.cost.total_tokens} tokens (${message.cost.input_tokens} in + ${message.cost.output_tokens} out)`}
                        >
                          <svg
                            className="w-3 h-3"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                          <span>${message.cost.cost_usd.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                  )}

                {/* Source Citations for Assistant Messages */}
                {message.role === "assistant" &&
                  !message.isError &&
                  (message.sources?.length > 0 ||
                    message.sections?.length > 0) && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <button
                        onClick={() => toggleSources(message.id)}
                        className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 transition-colors"
                      >
                        <svg
                          className={`w-4 h-4 transform transition-transform ${
                            expandedSources[message.id] ? "rotate-180" : ""
                          }`}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 9l-7 7-7-7"
                          />
                        </svg>
                        <span>
                          {expandedSources[message.id] ? "Hide" : "View"}{" "}
                          Sources (
                          {formatSources(message.sources, message.sections)
                            .uniqueSources.length +
                            formatSources(message.sources, message.sections)
                              .uniqueSections.length}
                          )
                        </span>
                      </button>

                      {expandedSources[message.id] && (
                        <div className="mt-3 space-y-3 animate-fadeIn">
                          {formatSources(message.sources, message.sections)
                            .uniqueSources.length > 0 && (
                            <div>
                              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                                Source Documents
                              </h4>
                              <ul className="space-y-1">
                                {formatSources(
                                  message.sources,
                                  message.sections,
                                ).uniqueSources.map((source, idx) => (
                                  <li
                                    key={idx}
                                    className="flex items-center gap-2 text-sm text-gray-700"
                                  >
                                    <svg
                                      className="w-4 h-4 text-gray-400 flex-shrink-0"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                      />
                                    </svg>
                                    <span className="truncate">{source}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {formatSources(message.sources, message.sections)
                            .uniqueSections.length > 0 && (
                            <div>
                              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                                Relevant Sections
                              </h4>
                              <ul className="space-y-1">
                                {formatSources(
                                  message.sources,
                                  message.sections,
                                ).uniqueSections.map((section, idx) => (
                                  <li
                                    key={idx}
                                    className="flex items-center gap-2 text-sm text-gray-700"
                                  >
                                    <svg
                                      className="w-4 h-4 text-gray-400 flex-shrink-0"
                                      fill="none"
                                      stroke="currentColor"
                                      viewBox="0 0 24 24"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14"
                                      />
                                    </svg>
                                    <span className="truncate">{section}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-900 border border-gray-200 rounded-2xl rounded-bl-md shadow-sm px-4 py-3">
                <div className="flex items-center gap-1">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Session Memory Panel */}
      {sessionMemory && sessionMemory.context_string && (
        <div className="bg-amber-50 border-t border-amber-200 px-4 py-2">
          <div className="max-w-3xl mx-auto">
            <button
              onClick={() => setIsMemoryExpanded(!isMemoryExpanded)}
              className="flex items-center gap-2 text-sm text-amber-700 hover:text-amber-900 transition-colors w-full"
            >
              <svg
                className="w-4 h-4 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
              <span className="font-medium">Session Memory</span>
              <svg
                className={`w-4 h-4 transition-transform ${isMemoryExpanded ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
              <span className="text-amber-600 text-xs ml-auto truncate max-w-[400px]">
                {!isMemoryExpanded &&
                  (sessionMemory.current_focus || sessionMemory.context_string)}
              </span>
            </button>

            {isMemoryExpanded && (
              <div className="mt-3 space-y-3 animate-fadeIn text-sm">
                {sessionMemory.user_profile && (
                  <div className="flex gap-2">
                    <span className="font-medium text-amber-800 whitespace-nowrap">
                      User:
                    </span>
                    <span className="text-amber-700">
                      {sessionMemory.user_profile}
                    </span>
                  </div>
                )}
                {sessionMemory.current_focus && (
                  <div className="flex gap-2">
                    <span className="font-medium text-amber-800 whitespace-nowrap">
                      Current Focus:
                    </span>
                    <span className="text-amber-700">
                      {sessionMemory.current_focus}
                    </span>
                  </div>
                )}
                {sessionMemory.conversation_summary && (
                  <div className="flex gap-2">
                    <span className="font-medium text-amber-800 whitespace-nowrap">
                      Summary:
                    </span>
                    <span className="text-amber-700">
                      {sessionMemory.conversation_summary}
                    </span>
                  </div>
                )}
                {sessionMemory.key_facts?.length > 0 && (
                  <div>
                    <span className="font-medium text-amber-800">
                      Key Facts:
                    </span>
                    <ul className="mt-1 ml-4 list-disc text-amber-700">
                      {sessionMemory.key_facts.map((fact, idx) => (
                        <li key={idx}>{fact}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {sessionMemory.anticipated_topics?.length > 0 && (
                  <div className="flex gap-2 flex-wrap">
                    <span className="font-medium text-amber-800 whitespace-nowrap">
                      Related Topics:
                    </span>
                    <span className="text-amber-700">
                      {sessionMemory.anticipated_topics.join(", ")}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about NovaTech..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <span>Send</span>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
