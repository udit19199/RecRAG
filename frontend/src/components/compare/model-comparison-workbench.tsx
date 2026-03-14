'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '@/components/ui/empty';
import { Separator } from '@/components/ui/separator';
import {
  getConfig,
  getProviders,
  type AdapterConfig,
  type ProviderInfo,
  type ProvidersResponse,
} from '@/lib/api';
import { cn } from '@/lib/utils';

type CompareRole = 'llm' | 'embedding';

type CompareModel = {
  id: string;
  provider: string;
  providerLabel: string;
  model: string;
  role: CompareRole;
  available: boolean;
  isActive: boolean;
};

const PROVIDER_LABELS: Record<string, string> = {
  ollama: 'Ollama',
  openai: 'OpenAI',
  nim: 'NVIDIA NIM',
};

const ROLE_LABELS: Record<CompareRole, string> = {
  llm: 'LLM',
  embedding: 'Embedding',
};

function makeValue(provider: string, model: string) {
  return `${provider}::${model}`;
}

function buildDefaults(
  role: CompareRole,
  providers: ProvidersResponse,
  config: { embedding: AdapterConfig; llm: AdapterConfig },
) {
  const providerMap = role === 'llm' ? providers.llms : providers.embedders;
  const active = role === 'llm' ? config.llm : config.embedding;
  const keys = Object.entries(providerMap).flatMap(([provider, info]) =>
    info.available ? info.models.map((model) => makeValue(provider, model)) : [],
  );

  const defaults: string[] = [];
  const activeKey = makeValue(active.provider, active.model);

  if (keys.includes(activeKey)) {
    defaults.push(activeKey);
  }

  for (const key of keys) {
    if (!defaults.includes(key)) {
      defaults.push(key);
    }

    if (defaults.length === 2) {
      break;
    }
  }

  return defaults;
}

function getModelFamily(model: string) {
  return model.split(/[:/]/)[0] || model;
}

function getModelVariant(model: string) {
  const parts = model.split(':');
  return parts.length > 1 ? parts.slice(1).join(':') : 'default tag';
}

function getActivationImpact(role: CompareRole, isActive: boolean) {
  if (isActive) {
    return 'Already active';
  }

  return role === 'embedding' ? 'Requires re-index' : 'Hot-swappable';
}

export function ModelComparisonWorkbench() {
  const [role, setRole] = useState<CompareRole>('llm');
  const [providers, setProviders] = useState<ProvidersResponse | null>(null);
  const [config, setConfig] = useState<{ embedding: AdapterConfig; llm: AdapterConfig } | null>(null);
  const [selectedByRole, setSelectedByRole] = useState<Record<CompareRole, string[]>>({
    llm: [],
    embedding: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [nextConfig, nextProviders] = await Promise.all([getConfig(), getProviders()]);
      setConfig(nextConfig);
      setProviders(nextProviders);
      setSelectedByRole({
        llm: buildDefaults('llm', nextProviders, nextConfig),
        embedding: buildDefaults('embedding', nextProviders, nextConfig),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load model inventory');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const providerMap = useMemo(() => {
    if (!providers) {
      return null;
    }

    return role === 'llm' ? providers.llms : providers.embedders;
  }, [providers, role]);

  const selectedModels = useMemo<CompareModel[]>(() => {
    if (!providerMap || !config) {
      return [];
    }

    const active = role === 'llm' ? config.llm : config.embedding;

    return selectedByRole[role]
      .map((value) => {
        const [provider, model] = value.split('::');
        if (!provider || !model) {
          return null;
        }

        const info = providerMap[provider];
        if (!info) {
          return null;
        }

        return {
          id: value,
          provider,
          providerLabel: PROVIDER_LABELS[provider] ?? provider,
          model,
          role,
          available: info.available,
          isActive: active.provider === provider && active.model === model,
        } satisfies CompareModel;
      })
      .filter((item): item is CompareModel => item !== null);
  }, [config, providerMap, role, selectedByRole]);

  const selectedCount = selectedByRole[role].length;
  const availableCount = useMemo(() => {
    if (!providerMap) {
      return 0;
    }

    return Object.values(providerMap).reduce((total, info) => {
      return total + (info.available ? info.models.length : 0);
    }, 0);
  }, [providerMap]);

  const availableProviders = useMemo(() => {
    if (!providerMap) {
      return 0;
    }

    return Object.values(providerMap).filter((info) => info.available).length;
  }, [providerMap]);

  const toggleSelection = (value: string) => {
    setSelectedByRole((current) => {
      const selection = current[role];

      if (selection.includes(value)) {
        return {
          ...current,
          [role]: selection.filter((entry) => entry !== value),
        };
      }

      if (selection.length >= 3) {
        return {
          ...current,
          [role]: [...selection.slice(1), value],
        };
      }

      return {
        ...current,
        [role]: [...selection, value],
      };
    });
  };

  return (
    <div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30 p-4">
      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="flex min-h-0 flex-col gap-4 rounded-xl border bg-card p-4">
          <div className="flex flex-col gap-2">
            <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
              Selection scope
            </p>
            <div className="grid grid-cols-2 gap-2">
              {(['llm', 'embedding'] as const).map((nextRole) => (
                <Button
                  key={nextRole}
                  type="button"
                  variant={role === nextRole ? 'default' : 'outline'}
                  onClick={() => setRole(nextRole)}
                >
                  {ROLE_LABELS[nextRole]}
                </Button>
              ))}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
            <SummaryCard label="Available models" value={availableCount} detail={`${availableProviders} providers online`} />
            <SummaryCard label="Selected" value={selectedCount} detail="Up to 3 side by side" />
            <SummaryCard
              label="Active profile"
              value={config ? ROLE_LABELS[role] : '--'}
              detail={config ? (role === 'llm' ? config.llm.model : config.embedding.model) : 'Waiting for API'}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-foreground">
                Candidate models
              </p>
              <p className="text-xs text-muted-foreground">
                Pick two or three entries to compare metadata and activation impact.
              </p>
            </div>
            <Button type="button" variant="outline" size="sm" onClick={loadData} disabled={isLoading}>
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </Button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            {error ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            ) : null}

            {!error && !providerMap ? (
              <div className="rounded-xl border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
                Loading available providers...
              </div>
            ) : null}

            {!error && providerMap ? (
              <div className="flex flex-col gap-4">
                {Object.entries(providerMap).map(([provider, info]) => (
                  <ProviderGroup
                    key={provider}
                    provider={provider}
                    info={info}
                    role={role}
                    selected={selectedByRole[role]}
                    onToggle={toggleSelection}
                  />
                ))}
              </div>
            ) : null}
          </div>
        </aside>

        <section className="flex min-h-0 flex-col gap-4">
          <div className="rounded-xl border bg-card px-5 py-4">
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                  Comparison board
                </p>
                <h2 className="text-lg font-semibold text-foreground">
                  {ROLE_LABELS[role]} inventory snapshot
                </h2>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {selectedModels.map((model) => (
                  <Badge key={model.id} variant={model.isActive ? 'default' : 'secondary'}>
                    {model.model}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {selectedModels.length >= 2 ? (
            <div className="grid min-h-0 flex-1 auto-rows-fr gap-4 md:grid-cols-2 xl:grid-cols-3">
              {selectedModels.map((model) => (
                <CompareCard key={model.id} model={model} />
              ))}
            </div>
          ) : (
            <Empty className="min-h-[320px] rounded-xl border bg-card">
              <EmptyHeader>
                <EmptyMedia variant="icon">//</EmptyMedia>
                <EmptyTitle>Select at least two models</EmptyTitle>
                <EmptyDescription>
                  Use the inventory list to choose multiple {ROLE_LABELS[role].toLowerCase()} candidates. The comparison grid updates instantly and keeps the current active model highlighted.
                </EmptyDescription>
              </EmptyHeader>
              <EmptyContent>
                <Badge variant="outline">Max 3 models</Badge>
              </EmptyContent>
            </Empty>
          )}
        </section>
      </div>
    </div>
  );
}

function SummaryCard({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return (
    <div className="rounded-xl border bg-muted/20 px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-lg font-semibold text-foreground">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function ProviderGroup({
  provider,
  info,
  role,
  selected,
  onToggle,
}: {
  provider: string;
  info: ProviderInfo;
  role: CompareRole;
  selected: string[];
  onToggle: (value: string) => void;
}) {
  const providerLabel = PROVIDER_LABELS[provider] ?? provider;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground">
          {providerLabel}
        </p>
        <Badge variant={info.available ? 'secondary' : 'outline'}>
          {info.available ? `${info.models.length} online` : 'Unavailable'}
        </Badge>
      </div>

      {!info.available ? (
        <div className="rounded-xl border border-dashed bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          {info.reason || `No ${ROLE_LABELS[role].toLowerCase()} models reported by this provider.`}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {info.models.map((model) => {
            const value = makeValue(provider, model);
            const isSelected = selected.includes(value);

            return (
              <button
                key={value}
                type="button"
                onClick={() => onToggle(value)}
                className={cn(
                  'flex w-full flex-col gap-1 rounded-xl border px-3 py-3 text-left transition-colors',
                  isSelected
                    ? 'border-foreground bg-muted text-foreground'
                    : 'bg-background hover:bg-muted/40',
                )}
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate text-sm font-medium text-foreground">{model}</span>
                  <Badge variant={isSelected ? 'default' : 'outline'}>
                    {isSelected ? 'Selected' : 'Add'}
                  </Badge>
                </div>
                <span className="text-xs text-muted-foreground">
                  {role === 'embedding' ? 'Vector indexing candidate' : 'Generation candidate'}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CompareCard({ model }: { model: CompareModel }) {
  return (
    <article className="flex h-full flex-col rounded-xl border bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
            {ROLE_LABELS[model.role]}
          </p>
          <h3 className="mt-2 break-words text-base font-semibold text-foreground">
            {model.model}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">{model.providerLabel}</p>
        </div>

        <div className="flex flex-col items-end gap-2">
          <Badge variant={model.isActive ? 'default' : 'secondary'}>
            {model.isActive ? 'Active' : 'Candidate'}
          </Badge>
          <Badge variant={model.available ? 'outline' : 'destructive'}>
            {model.available ? 'Available' : 'Offline'}
          </Badge>
        </div>
      </div>

      <Separator className="my-4" />

      <dl className="flex flex-1 flex-col gap-3 text-sm">
        <MetricRow label="Provider" value={model.providerLabel} />
        <MetricRow label="Family" value={getModelFamily(model.model)} />
        <MetricRow label="Variant" value={getModelVariant(model.model)} />
        <MetricRow label="Switch impact" value={getActivationImpact(model.role, model.isActive)} />
        <MetricRow
          label="Current role"
          value={model.role === 'embedding' ? 'Retrieval embeddings' : 'Response generation'}
        />
      </dl>
    </article>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-dashed pb-3 last:border-b-0 last:pb-0">
      <dt className="text-xs uppercase tracking-[0.14em] text-muted-foreground">{label}</dt>
      <dd className="max-w-[65%] text-right text-sm text-foreground">{value}</dd>
    </div>
  );
}
