'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import FileUploader from '@/components/FileUploader';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
import ModelPicker from '@/components/ModelPicker';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import Sidebar, { SidebarProvider, SidebarHeader, SidebarContent, SidebarFooter, SidebarGroup } from '@/components/ui/sidebar';
import { Switch } from '@/components/ui/switch';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  queryRAG,
  queryRAGWithEval,
  getEvalStatus,
  checkRetrievalHealth,
  uploadPDFs,
  getIngestionStatus,
  waitForIngestionComplete,
  type QueryResponse,
  type QueryWithEvalResponse,
  type IngestionStatus,
} from '@/lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  context?: QueryResponse['context'];
  error?: string;
  eval?: { [metric: string]: number } | undefined;
  eval_job_id?: string;
  eval_status?: 'pending' | 'complete' | 'error';
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [isReady, setIsReady] = useState(false);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);

  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [enableEval, setEnableEval] = useState(false);
  const [evalMode, setEvalMode] = useState<'async' | 'sync'>('async');
  const [isQuerying, setIsQuerying] = useState(false);
  const [messageIdCounter, setMessageIdCounter] = useState(0);
  

  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Restore persisted prefs
  useEffect(() => {
    try {
      const stored = localStorage.getItem('enableEval');
      if (stored !== null) setEnableEval(stored === 'true');
      const storedMode = localStorage.getItem('evalMode');
      if (storedMode === 'sync' || storedMode === 'async') setEvalMode(storedMode as 'async' | 'sync');
    } catch { /* ignore */ }
  }, []);

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
        setHealthMessage(health.pipeline_loaded ? null : 'Retrieval pipeline is loading…');
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

  const handleReindexStarted = useCallback(() => {
    fetchIngestionStatus();
  }, [fetchIngestionStatus]);

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
      const result = enableEval ? await queryRAGWithEval(query, evalMode) : await queryRAG(query);
      const asyncEvalJobId = enableEval && evalMode === 'async'
        ? (result as QueryWithEvalResponse).eval_job_id
        : undefined;
      const assistantMsg: Message = {
        id: nextId(),
        role: 'assistant',
        content: result.response,
        context: result.context,
      };
      if (asyncEvalJobId) {
        assistantMsg.eval_job_id = asyncEvalJobId;
        assistantMsg.eval_status = 'pending';
      }
      setMessages((prev) => [...prev, assistantMsg]);

      if (asyncEvalJobId) {
        const jobId = asyncEvalJobId;
        (async function poll() {
          try {
            for (;;) {
              const status = await getEvalStatus(jobId);
              if (status.status === 'complete') {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.role === 'assistant' && m.eval_job_id === jobId
                      ? { ...m, ...(status.scores ? { eval: status.scores } : {}), eval_status: 'complete' }
                      : m
                  )
                );
                break;
              }
              if (status.status === 'error') break;
              await new Promise((r) => setTimeout(r, 1500));
            }
          } catch { /* ignore polling errors */ }
        })();
      }
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
      await uploadPDFs(files);
      setUploadFeedback({
        type: 'success',
        message: `${files.length} file${files.length > 1 ? 's' : ''} uploaded. Existing corpus replaced and processing started.`,
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
  const canChat = isReady && ingestionStatus?.status === 'complete' && !isIngesting;

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full flex-col overflow-hidden bg-muted/30">
        {/* header removed - controls moved into main column */}

        <div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
          <Sidebar className="">
            <SidebarHeader>
              <div className="space-y-1">
                <h2 className="text-sm font-semibold text-foreground">Library</h2>
                <p className="text-xs text-muted-foreground">Upload your full PDF set, then wait for indexing to finish.</p>
              </div>
            </SidebarHeader>
            <SidebarContent>
              <SidebarGroup>
                <IngestionStatusDisplay status={ingestionStatus} error={statusError} isLoading={false} />
              </SidebarGroup>
              <SidebarGroup>
                {uploadFeedback && (
                  <div className={`rounded-lg border px-3 py-2.5 text-xs ${uploadFeedback.type === 'success' ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400' : 'border-destructive/30 bg-destructive/5 text-destructive'}`}>
                    {uploadFeedback.message}
                  </div>
                )}
                <FileUploader onUpload={handleUpload} isUploading={isUploading} disabled={isIngesting} />
              </SidebarGroup>
              <SidebarGroup>
                <div className="space-y-3 rounded-xl border bg-card p-4">
                  <h3 className="text-xs font-semibold text-foreground">Tips</h3>
                  <div className="space-y-2">
                    <TipRow text="Use text-based PDFs, not scanned images" />
                    <TipRow text="Keep files under 50 MB" />
                    <TipRow text="Wait for Complete before querying" />
                    <TipRow text="A new upload batch replaces the current corpus" />
                  </div>
                </div>
              </SidebarGroup>
            </SidebarContent>
            <SidebarFooter>
              <div className="text-xs text-muted-foreground">Status: {ingestionStatus?.status ?? 'idle'}</div>
            </SidebarFooter>
          </Sidebar>

          <section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
            <div className="flex-1 overflow-hidden rounded-xl border bg-card flex flex-col">
              <div className="px-4 py-3 border-b bg-background">
                <div className="flex flex-wrap items-center gap-3">
                  <ModelPicker disabled={isIngesting} onReindexStarted={handleReindexStarted} onConfigChanged={() => {}} />
                  <div className="hidden h-8 w-px bg-border md:block" />
                  <label className="flex items-center gap-2 text-sm text-muted-foreground">
                    <span>Evaluation</span>
                    <Switch
                      checked={enableEval}
                      onCheckedChange={(checked) => {
                        setEnableEval(checked as boolean);
                        try { localStorage.setItem('enableEval', String(checked)); } catch {}
                      }}
                    />
                  </label>

                  {enableEval && (
                    <Select
                      value={evalMode}
                      onValueChange={(v) => {
                        setEvalMode(v as 'async' | 'sync');
                        try { localStorage.setItem('evalMode', v); } catch {}
                      }}
                    >
                      <SelectTrigger className="h-9 w-32 bg-background text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="async">Async</SelectItem>
                        <SelectItem value="sync">Sync</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                </div>
              </div>

              <div className="h-full overflow-y-auto p-4 md:p-6">
                {messages.length === 0 ? (
                  <EmptyState isReady={isReady} />
                ) : (
                  <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
                    {messages.map((msg) => (
                      <ChatMessage key={msg.id} message={msg} />
                    ))}
                    {isQuerying && <TypingIndicator />}
                    <div ref={chatBottomRef} />
                  </div>
                )}
              </div>
            </div>

          <div className="shrink-0 rounded-xl border bg-card p-4">
            <ChatInput
              onSubmit={handleQuery}
              isLoading={isQuerying}
              disabled={!canChat}
              disabledReason={
                isIngesting
                  ? 'Document indexing in progress…'
                  : !isReady
                  ? 'Waiting for retrieval pipeline…'
                  : ingestionStatus?.status !== 'complete'
                  ? 'Upload and index your full document set before chatting.'
                  : undefined
              }
            />
          </div>
        </section>
      </div>
    </div>
    </SidebarProvider>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function EmptyState({ isReady }: { isReady: boolean }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="flex w-full max-w-xl flex-col items-center gap-4 rounded-xl border border-dashed bg-muted/30 px-6 py-12 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg border bg-background">
          <svg className="h-6 w-6 text-muted-foreground" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
          </svg>
        </div>
        <div>
          <p className="font-semibold text-foreground">
            {isReady ? 'Ask a question about your documents' : 'Pipeline loading…'}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {isReady
              ? 'Upload your full PDF batch in the library panel. Chat unlocks after indexing completes.'
              : 'The retrieval service is initialising. Please wait a moment.'}
          </p>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-[11px] font-medium text-muted-foreground">
        AI
      </div>
      <div className="rounded-xl border bg-muted/30 px-4 py-3">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"
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
        <div className="max-w-[75%] rounded-xl border bg-muted px-4 py-3 text-sm text-foreground">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-[11px] font-medium text-muted-foreground">
        AI
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {message.error ? (
          <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {message.error}
          </div>
        ) : (
          <div className="rounded-xl border bg-background px-4 py-3">
            <div className="prose prose-sm max-w-none text-foreground [&_a]:text-primary [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-sm [&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_li]:text-foreground [&_ol]:text-foreground [&_p]:text-foreground [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-4 [&_strong]:text-foreground [&_ul]:text-foreground">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {message.context && message.context.length > 0 && (
          <SourceContext context={message.context} />
        )}

        {message.eval && (
          <div className="rounded-xl border bg-background px-4 py-3">
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              Evaluation
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(message.eval).map(([k, v]) => (
                <Badge key={k} variant="secondary" className="gap-2">
                  <span>{k}</span>
                  <span className="text-muted-foreground">{v.toFixed(2)}</span>
                </Badge>
              ))}
            </div>
          </div>
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
      if (next.has(i)) next.delete(i); else next.add(i);
      return next;
    });
  };

  return (
    <div className="overflow-hidden rounded-xl border bg-background">
      <div className="flex items-center gap-2 border-b px-4 py-2.5">
        <span className="text-sm font-semibold text-foreground">Sources</span>
        <Badge variant="secondary" className="h-4 min-w-4 px-1 text-[10px]">
          {context.length}
        </Badge>
      </div>
      <div className="divide-y divide-border">
        {context.map((item, i) => (
          <Collapsible
            key={i}
            open={openIndexes.has(i)}
            onOpenChange={() => toggle(i)}
          >
            <CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-2.5 text-left transition-colors hover:bg-muted/50">
              <div className="flex min-w-0 items-center gap-2.5">
                <Badge variant="outline" className="h-5 w-5 rounded-full p-0 text-[10px]">
                  {i + 1}
                </Badge>
                <span className="truncate text-xs font-medium text-foreground">
                  {item.source}
                </span>
              </div>
              <div className="ml-3 flex shrink-0 items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="cursor-help text-[10px] text-muted-foreground">
                      {item.distance.toFixed(4)}
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Vector similarity score</p>
                  </TooltipContent>
                </Tooltip>
                <svg
                  className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${openIndexes.has(i) ? 'rotate-180' : ''}`}
                  viewBox="0 0 20 20" fill="currentColor"
                >
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="border-t bg-muted/40 px-4 py-3">
                <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
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
    <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
      {disabledReason && (
        <p className="mb-2 text-center text-xs text-muted-foreground">{disabledReason}</p>
      )}
      <div className="flex items-end gap-3 rounded-xl border bg-background px-4 py-3 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/20">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          rows={1}
          disabled={disabled}
          className="max-h-36 min-h-[1.5rem] flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed"
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = `${el.scrollHeight}px`;
          }}
        />
        <Button
          type="submit"
          disabled={!value.trim() || isLoading || disabled}
          size="icon"
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
        </Button>
      </div>
      <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
        Enter to send, Shift+Enter for newline
      </p>
    </form>
  );
}

function TipRow({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2 text-xs text-muted-foreground">
      <svg className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M16.704 5.29a1 1 0 010 1.414l-7.02 7.02a1 1 0 01-1.414 0l-3.02-3.02a1 1 0 111.414-1.414l2.313 2.313 6.313-6.313a1 1 0 011.414 0z" clipRule="evenodd" />
      </svg>
      <span>{text}</span>
    </div>
  );
}
