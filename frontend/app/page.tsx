'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ModelPicker from '@/components/ModelPicker';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
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
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { UploadDialog } from '@/components/UploadDialog';
import { IconCloud, IconRobot } from '@tabler/icons-react';
import {
  queryRAG,
  getEvalStatus,
  checkRetrievalHealth,
  uploadPDFs,
  getIngestionStatus,
  waitForIngestionComplete,
  type QueryResponse,
  type IngestionStatus,
} from '@/lib/api';
import { cn } from '@/lib/utils';

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

  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [messageIdCounter, setMessageIdCounter] = useState(0);


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
      } catch {
        setIsReady(false);
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
  });

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
      const asyncEvalJobId = result.eval_job_id;
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
            for (; ;) {
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
    <div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30">
      <div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
        <section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
          <div className="flex-1 overflow-hidden rounded-xl border bg-card flex flex-col">
            <div className="px-4 py-3 border-b bg-background space-y-3">
              {isIngesting || (uploadFeedback && uploadFeedback.type === 'error') ? (
                <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                  <IngestionStatusDisplay status={ingestionStatus} error={statusError} isLoading={false} />
                  {uploadFeedback && uploadFeedback.type === 'error' && (
                    <div className="mt-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                      {uploadFeedback.message}
                    </div>
                  )}
                </div>
              ) : null}

              {uploadFeedback && uploadFeedback.type === 'success' && !isIngesting && (
                <div className="animate-in fade-in slide-in-from-top-1 duration-300 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-600 dark:text-emerald-400">
                  {uploadFeedback.message}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-3">
                <ModelPicker disabled={isIngesting} onReindexStarted={handleReindexStarted} onConfigChanged={() => { }} />
              </div>
            </div>
            <div className="h-full overflow-y-auto p-4 md:p-6">
              {messages.length === 0 ? (
                <EmptyState isReady={isReady} onUpload={handleUpload} isUploading={isUploading} />
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
              onUpload={handleUpload}
              isLoading={isQuerying}
              isUploading={isUploading || isIngesting}
              disabled={!canChat || isIngesting}
              disabledReason={
                isIngesting
                  ? 'Document indexing in progress…'
                  : !isReady
                    ? 'Waiting for retrieval pipeline…'
                    : ingestionStatus?.status !== 'complete'
                      ? 'Upload and index your documents before chatting.'
                      : undefined
              }
            />
          </div>
        </section>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function EmptyState({ isReady, onUpload, isUploading }: { isReady: boolean; onUpload: (files: File[]) => Promise<void>; isUploading: boolean }) {
  return (
    <div className="flex h-full items-center justify-center">
      <Empty className="border border-dashed bg-muted/20 rounded-2xl max-w-md w-full py-12">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <IconCloud className="h-6 w-6 text-muted-foreground" />
          </EmptyMedia>
          <EmptyTitle>{isReady ? 'Library Empty' : 'Pipeline Loading…'}</EmptyTitle>
          <EmptyDescription>
            {isReady
              ? 'Upload your PDF documents to the library to start chatting with your AI.'
              : 'The retrieval service is initialising. Please wait a moment.'}
          </EmptyDescription>
        </EmptyHeader>
        {isReady && (
          <EmptyContent>
            <UploadDialog onUpload={onUpload} isUploading={isUploading} />
          </EmptyContent>
        )}
      </Empty>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground">
        <IconRobot className="h-4 w-4" />
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
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground">
        <IconRobot className="h-4 w-4" />
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
            key={`${item.source}-${item.distance}-${item.text}`}
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
                  aria-hidden="true"
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
  onUpload,
  isLoading,
  isUploading,
  disabled,
  disabledReason,
}: {
  onSubmit: (query: string) => Promise<void>;
  onUpload: (files: File[]) => Promise<void>;
  isLoading: boolean;
  isUploading: boolean;
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
      <div
        className={cn(
          "flex items-end gap-3 rounded-xl border bg-background px-4 py-3 transition-all",
          !disabled && "focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/20",
          disabled && "cursor-not-allowed bg-muted/20 opacity-60"
        )}
      >
        <UploadDialog
          onUpload={onUpload}
          isUploading={isUploading}
          trigger={
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-9 w-9 shrink-0 text-muted-foreground hover:text-foreground"
              disabled={isUploading}
              title="Upload PDFs"
            >
              {isUploading ? (
                <svg aria-hidden="true" className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              ) : (
                <svg aria-hidden="true" className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
              )}
            </Button>
          }
        />
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? (disabledReason || "Chat disabled") : "Ask a question about your documents..."}
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
            <svg aria-hidden="true" className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg aria-hidden="true" className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
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
