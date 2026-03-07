'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxInput,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
} from '@/components/ui/combobox';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  getConfig,
  getProviders,
  setConfig,
  setIngestionConfig,
  triggerReindex,
  type AdapterConfig,
  type ProvidersResponse,
} from '@/lib/api';

interface ModelPickerProps {
  /** Called after an embedding model change that requires re-indexing */
  onReindexStarted?: () => void;
  /** Called after a successful LLM or embedding model change */
  onConfigChanged?: (config: { embedding: AdapterConfig; llm: AdapterConfig }) => void;
  disabled?: boolean;
}

type PendingEmbedChange = { provider: string; model: string } | null;

const PROVIDER_LABELS: Record<string, string> = {
  ollama: 'Ollama',
  openai: 'OpenAI',
  nim: 'NVIDIA NIM',
};

export default function ModelPicker({
  onReindexStarted,
  onConfigChanged,
  disabled = false,
}: ModelPickerProps) {
  const [providers, setProviders] = useState<ProvidersResponse | null>(null);
  const [currentEmbedding, setCurrentEmbedding] = useState<AdapterConfig | null>(null);
  const [currentLLM, setCurrentLLM] = useState<AdapterConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [llmChanging, setLlmChanging] = useState(false);
  const [embedChanging, setEmbedChanging] = useState(false);
  const [pendingEmbedChange, setPendingEmbedChange] = useState<PendingEmbedChange>(null);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [cfg, prov] = await Promise.all([getConfig(), getProviders()]);
      setCurrentEmbedding(cfg.embedding);
      setCurrentLLM(cfg.llm);
      setProviders(prov);
    } catch {
      // silently ignore — dropdowns will stay as skeletons if APIs are down
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Load persisted selections from localStorage (fast UI) — will be overwritten by server config
  useEffect(() => {
    try {
      const llm = localStorage.getItem('lastLLM');
      const emb = localStorage.getItem('lastEmbedding');
      if (llm && !currentLLM) {
        const [provider, model] = llm.split('::');
        if (provider && model) setCurrentLLM({ provider, model });
      }
      if (emb && !currentEmbedding) {
        const [provider, model] = emb.split('::');
        if (provider && model) setCurrentEmbedding({ provider, model });
      }
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── LLM change ──────────────────────────────────────────────────────────────
  const handleLLMChange = async (value: string) => {
    // value format: "provider::model"
    const [provider, model] = value.split('::');
    if (!provider || !model) return;
    if (provider === currentLLM?.provider && model === currentLLM?.model) return;

    setLlmChanging(true);
    try {
      const result = await setConfig({ llm: { provider, model } });
      setCurrentLLM(result.llm);
      try { localStorage.setItem('lastLLM', makeValue(result.llm.provider, result.llm.model)); } catch {}
      onConfigChanged?.({ embedding: result.embedding, llm: result.llm });
    } catch (err) {
      console.error('Failed to change LLM:', err);
    } finally {
      setLlmChanging(false);
    }
  };

  // ── Embedding change (two-step with confirmation dialog) ───────────────────
  const handleEmbedChange = (value: string) => {
    const [provider, model] = value.split('::');
    if (!provider || !model) return;
    if (provider === currentEmbedding?.provider && model === currentEmbedding?.model) return;
    setPendingEmbedChange({ provider, model });
  };

  const confirmEmbedChange = async () => {
    if (!pendingEmbedChange) return;
    const { provider, model } = pendingEmbedChange;
    setPendingEmbedChange(null);
      setEmbedChanging(true);

    try {
      // Update both retrieval and ingestion pipelines
      const [result] = await Promise.all([
        setConfig({ embedding: { provider, model } }),
        setIngestionConfig({ provider, model }),
      ]);

      setCurrentEmbedding(result.embedding);
      try { localStorage.setItem('lastEmbedding', makeValue(result.embedding.provider, result.embedding.model)); } catch {}
      onConfigChanged?.({ embedding: result.embedding, llm: result.llm });

      if (result.requires_reindex) {
        await triggerReindex();
        onReindexStarted?.();
      }
    } catch (err) {
      console.error('Failed to change embedding model:', err);
    } finally {
      setEmbedChanging(false);
    }
  };

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const makeValue = (provider: string, model: string) => `${provider}::${model}`;

  const currentLLMValue =
    currentLLM ? makeValue(currentLLM.provider, currentLLM.model) : undefined;
  const currentEmbedValue =
    currentEmbedding ? makeValue(currentEmbedding.provider, currentEmbedding.model) : undefined;

  const renderSelect = (
    role: 'llm' | 'embedding',
    label: string,
    value: string | undefined,
    onChange: (v: string) => void,
    isChanging: boolean
  ) => {
    if (isLoading || !providers) {
      return (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-muted-foreground">
            {label}
          </span>
          <Skeleton className="h-10 w-[13.5rem] rounded-md" />
        </div>
      );
    }

    const providerMap = role === 'llm' ? providers.llms : providers.embedders;

    return (
      <div className="flex flex-col gap-1.5">
        <span className="text-xs text-muted-foreground">
          {label}
        </span>
        <Combobox
          items={Object.entries(providerMap).flatMap(([providerKey, info]) =>
            info.available ? info.models.map((model) => makeValue(providerKey, model)) : []
          )}
          value={value ?? null}
          onValueChange={(nextValue) => {
            if (typeof nextValue === 'string') onChange(nextValue);
          }}
          disabled={disabled || isChanging}
        >
          <ComboboxInput
            placeholder={isChanging ? 'Applying...' : 'Select model'}
            readOnly={isChanging}
            className={`w-[13.5rem] ${isChanging ? 'opacity-60' : ''}`}
          />
          <ComboboxContent>
            <ComboboxEmpty>No matching models.</ComboboxEmpty>
            <ComboboxList>
            {Object.entries(providerMap).map(([providerKey, info]) => {
              const label = PROVIDER_LABELS[providerKey] ?? providerKey;
              if (!info.available || info.models.length === 0) {
                return (
                  <ComboboxGroup key={providerKey}>
                    <ComboboxLabel className="flex items-center justify-between">
                      <span>{label}</span>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="ml-2 cursor-default text-xs text-muted-foreground">unavailable</span>
                        </TooltipTrigger>
                        <TooltipContent side="right">
                          <p className="max-w-xs text-xs">
                            {info.reason || 'Provider not available'}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </ComboboxLabel>
                  </ComboboxGroup>
                );
              }
              return (
                <ComboboxGroup key={providerKey}>
                  <ComboboxLabel>{label}</ComboboxLabel>
                  {info.models.map((model) => (
                    <ComboboxItem
                      key={makeValue(providerKey, model)}
                      value={makeValue(providerKey, model)}
                    >
                      {model}
                    </ComboboxItem>
                  ))}
                </ComboboxGroup>
              );
            })}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
      </div>
    );
  };

  return (
    <>
      <div className="flex items-end gap-4">
        {renderSelect(
          'embedding',
          'Embedding',
          currentEmbedValue,
          handleEmbedChange,
          embedChanging
        )}
        {renderSelect(
          'llm',
          'LLM',
          currentLLMValue,
          handleLLMChange,
          llmChanging
        )}
      </div>

      {/* Re-index confirmation dialog */}
      <AlertDialog
        open={pendingEmbedChange !== null}
        onOpenChange={(open) => { if (!open) setPendingEmbedChange(null); }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Change embedding model?</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                Switching to{' '}
                <span className="font-mono font-semibold">
                  {pendingEmbedChange?.model}
                </span>{' '}
                will require re-indexing all your documents.
              </span>
              <span className="block text-muted-foreground">
                Existing search will be unavailable until re-indexing completes. This may take several minutes depending on the size of your document collection.
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmEmbedChange}>
              Change &amp; Re-index
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
