'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import FileUploader from '@/components/FileUploader';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
import ModelPicker from '@/components/ModelPicker';
import { Separator } from '@/components/ui/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  queryRAG,
  checkRetrievalHealth,
  uploadPDF,
  getIngestionStatus,
  waitForIngestionComplete,
  type QueryResponse,
  type IngestionStatus,
  type AdapterConfig,
} from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  context?: QueryResponse['context'];
  error?: string;
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  // Pipeline readiness
  const [isReady, setIsReady] = useState(false);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);

  // Ingestion state
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [messageIdCounter, setMessageIdCounter] = useState(0);

  // Scroll ref
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // ── Ingestion status polling ─────────────────────────────────────────────

  const fetchIngestionStatus = useCallback(async () => {
    try {
      const s = await getIngestionStatus();
      setIngestionStatus(s);
      setStatusError(null);
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, []);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await checkRetrievalHealth();
        setIsReady(health.pipeline_loaded ?? false);
        if (!health.pipeline_loaded) {
          setHealthMessage('Retrieval pipeline is loading…');
        } else {
          setHealthMessage(null);
        }
      } catch {
        setHealthMessage('Cannot reach retrieval API. Ensure the service is running.');
      }
    };

    checkHealth();
    fetchIngestionStatus();
  }, [fetchIngestionStatus]);

  useEffect(() => {
    if (ingestionStatus?.status !== 'processing') return;
    const interval = setInterval(fetchIngestionStatus, 3000);
    return () => clearInterval(interval);
  }, [ingestionStatus?.status, fetchIngestionStatus]);

  // After a re-index starts, begin polling immediately
  const handleReindexStarted = useCallback(() => {
    fetchIngestionStatus();
  }, [fetchIngestionStatus]);

  // ── Auto-scroll on new messages ──────────────────────────────────────────

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Query handler ────────────────────────────────────────────────────────

  const nextId = () => {
    const id = messageIdCounter;
    setMessageIdCounter((c) => c + 1);
    return id;
  };

  const handleQuery = async (query: string) => {
    if (!query.trim() || isQuerying) return;

    const userMsg: Message = { id: nextId(), role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setIsQuerying(true);

    try {
      const result = await queryRAG(query);
      const assistantMsg: Message = {
        id: nextId(),
        role: 'assistant',
        content: result.response,
        context: result.context,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: nextId(),
        role: 'assistant',
        content: '',
        error: err instanceof Error ? err.message : 'An unexpected error occurred',
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  // ── Upload handler ───────────────────────────────────────────────────────

  const handleUpload = async (files: File[]) => {
    setIsUploading(true);
    setUploadFeedback(null);

    try {
      for (const file of files) {
        await uploadPDF(file);
      }
      setUploadFeedback({
        type: 'success',
        message: `${files.length} file${files.length > 1 ? 's' : ''} uploaded. Processing started.`,
      });
      const finalStatus = await waitForIngestionComplete();
      setIngestionStatus(finalStatus);
    } catch (err) {
      setUploadFeedback({
        type: 'error',
        message: err instanceof Error ? err.message : 'Upload failed',
      });
    } finally {
      setIsUploading(false);
    }
  };

  const isIngesting = ingestionStatus?.status === 'processing';

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex h-[calc(100vh-65px)] w-full flex-col">
      {/* ── Page header with model pickers ──────────────────────────────── */}
          <div className="flex items-center justify-between border-b border-transparent bg-gradient-to-b from-zinc-950/80 to-transparent px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold text-zinc-50">
            Chat
          </h1>
          {healthMessage && (
            <p className="text-xs text-amber-400">{healthMessage}</p>
          )}
        </div>
          <div className="flex items-center gap-3">
            <ModelPicker
              disabled={isIngesting}
              onReindexStarted={handleReindexStarted}
              onConfigChanged={(_cfg: { embedding: AdapterConfig; llm: AdapterConfig }) => {
                // optionally handle config changes
              }}
            />
          </div>
        </div>

      {/* ── Main two-column layout ───────────────────────────────────────── */}
      <div className="flex min-h-0 flex-1">
        {/* ── Left sidebar: Documents ──────────────────────────────────── */}
        <aside className="flex w-72 md:w-80 lg:w-96 shrink-0 flex-col gap-4 overflow-y-auto border-r subtle-divider bg-zinc-950 p-5">
          <div>
            <h2 className="mb-1 text-sm font-semibold text-zinc-50">
              Documents
            </h2>
            <p className="text-xs text-zinc-400">
              Upload PDFs to be indexed and searchable.
            </p>
          </div>

          {/* Ingestion status */}
          <IngestionStatusDisplay
            status={ingestionStatus}
            error={statusError}
            isLoading={false}
          />

          {/* Upload feedback */}
          {uploadFeedback && (
              <div
                className={`flex items-start gap-2 rounded-lg p-3 text-xs ${
                  uploadFeedback.type === 'success'
                    ? 'bg-green-950/40 text-green-300 soft-border'
                    : 'bg-red-950 text-red-300 soft-border'
                }`}
              >
              {uploadFeedback.type === 'success' ? (
                <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="mt-0.5 h-4 w-4 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              )}
              <span>{uploadFeedback.message}</span>
            </div>
          )}

          <FileUploader
            onUpload={handleUpload}
            isUploading={isUploading}
            disabled={isIngesting}
          />

          <Separator />

          {/* Tips */}
          <div>
            <h3 className="mb-2 text-xs font-medium text-zinc-300">
              Tips for better results
            </h3>
            <ul className="space-y-1.5 text-xs text-zinc-400">
              <li>• Use text-based PDFs, not scanned images</li>
              <li>• Keep files under 50 MB</li>
              <li>• Wait for &quot;Complete&quot; before querying</li>
              <li>• Multiple files can be uploaded at once</li>
            </ul>
          </div>
        </aside>

        {/* ── Right panel: Chat ─────────────────────────────────────────── */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Chat thread */}
          <div className="flex-1 overflow-y-auto bg-zinc-950 px-6 py-6">
            {messages.length === 0 ? (
              <EmptyState isReady={isReady} />
            ) : (
              <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}
                {isQuerying && <TypingIndicator />}
                <div ref={chatBottomRef} />
              </div>
            )}
            {messages.length > 0 && !isQuerying && (
              <div ref={chatBottomRef} />
            )}
          </div>

          {/* Input bar */}
          <div className="border-t subtle-divider bg-zinc-950 px-6 py-4">
            <ChatInput
              onSubmit={handleQuery}
              isLoading={isQuerying}
              disabled={!isReady || isIngesting}
              disabledReason={
                isIngesting
                  ? 'Document indexing in progress…'
                  : !isReady
                  ? 'Waiting for retrieval pipeline…'
                  : undefined
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function EmptyState({ isReady }: { isReady: boolean }) {
  return (
    <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-3 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-zinc-800">
        <svg className="h-7 w-7 text-zinc-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
        </svg>
      </div>
      <div>
        <p className="font-medium text-zinc-300">
            {isReady ? 'Ask a question about your documents' : 'Pipeline loading…'}
          </p>
        <p className="mt-1 text-sm text-zinc-400">
          {isReady
            ? 'Upload PDFs on the left, then type your question below.'
            : 'The retrieval service is initialising. Please wait a moment.'}
        </p>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-2">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white text-xs font-bold">
        AI
      </div>
        <div className="rounded-2xl rounded-tl-sm bg-zinc-800 px-4 py-3 shadow-sm">
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 animate-bounce rounded-full bg-zinc-400"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-blue-600 px-4 py-3 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white text-xs font-bold">
        AI
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {message.error ? (
          <div className="rounded-2xl rounded-tl-sm bg-red-950 px-4 py-3 text-sm text-red-300 soft-border">
            {/* use soft-border utility for less harsh visuals */}
            {message.error}
          </div>
        ) : (
          <div className="rounded-2xl rounded-tl-sm bg-zinc-800 px-4 py-3 shadow-sm">
            <div className="prose prose-sm prose-zinc max-w-none prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Source context accordion */}
        {message.context && message.context.length > 0 && (
          <SourceContext context={message.context} />
        )}
      </div>
    </div>
  );
}

function SourceContext({ context }: { context: QueryResponse['context'] }) {
  const [openIndexes, setOpenIndexes] = useState<Set<number>>(new Set());

  const toggle = (i: number) => {
    setOpenIndexes((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  return (
          <div className="rounded-xl soft-border bg-zinc-900">
      <div className="border-b subtle-divider px-4 py-2.5">
        <span className="text-xs font-medium text-zinc-400">
          {context.length} source{context.length > 1 ? 's' : ''} retrieved
        </span>
      </div>
      <div className="divide-y divide-zinc-700">
        {context.map((item, i) => (
          <Collapsible
            key={i}
            open={openIndexes.has(i)}
            onOpenChange={() => toggle(i)}
          >
            <CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-2.5 text-left hover:bg-zinc-700">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-700 text-[10px] font-semibold text-zinc-300">
                  {i + 1}
                </span>
                <span className="truncate text-xs font-medium text-zinc-300">
                  {item.source}
                </span>
              </div>
              <div className="ml-3 flex shrink-0 items-center gap-2">
                <span className="text-[10px] text-zinc-400">
                  {item.distance.toFixed(4)}
                </span>
                <svg
                  className={`h-3.5 w-3.5 text-zinc-400 transition-transform ${openIndexes.has(i) ? 'rotate-180' : ''}`}
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="border-t subtle-divider bg-zinc-900/40 px-4 py-3">
                <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-zinc-400">
                  {item.text}
                </p>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </div>
    </div>
  );
}

function ChatInput({
  onSubmit,
  isLoading,
  disabled,
  disabledReason,
}: {
  onSubmit: (query: string) => Promise<void>;
  isLoading: boolean;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const [value, setValue] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || isLoading || disabled) return;
    const query = value;
    setValue('');
    await onSubmit(query);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
      {disabledReason && (
        <p className="mb-2 text-center text-xs text-zinc-400">{disabledReason}</p>
      )}
          <div className="flex items-end gap-3 rounded-xl soft-border bg-zinc-800 px-4 py-3 focus-within:ring-2 focus-within:ring-blue-500/20">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for newline)"
          rows={1}
          className="max-h-36 min-h-[1.5rem] flex-1 resize-none bg-transparent text-sm text-zinc-50 placeholder-zinc-500 focus:outline-none disabled:cursor-not-allowed"
          disabled={disabled}
          style={{
            height: 'auto',
          }}
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = `${el.scrollHeight}px`;
          }}
        />
        <button
          type="submit"
          disabled={!value.trim() || isLoading || disabled}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isLoading ? (
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          )}
        </button>
      </div>
    </form>
  );
}
