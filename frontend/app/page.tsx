'use client';

import { useState, useEffect } from 'react';
import QueryForm from '@/components/QueryForm';
import QueryResults from '@/components/QueryResults';
import {
  queryRAG,
  checkRetrievalHealth,
  type QueryResponse,
} from '@/lib/api';

export default function QueryPage() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await checkRetrievalHealth();
        setIsReady(health.pipeline_loaded ?? false);
        if (!health.pipeline_loaded) {
          setHealthError('Retrieval pipeline is still loading. Please wait a moment.');
        }
      } catch (err) {
        setHealthError('Unable to connect to retrieval API. Please ensure the service is running.');
      }
    };

    checkHealth();
  }, []);

  const handleQuery = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await queryRAG(query);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          Query Documents
        </h1>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">
          Ask questions about your uploaded documents and get AI-powered answers
          with source citations.
        </p>
      </div>

      {/* Health status */}
      {!isReady && !isLoading && (
        <div className="flex items-center gap-3 py-4">
          <svg
            className="h-5 w-5 text-yellow-500"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
          <p className="text-yellow-600 dark:text-yellow-400">
            {healthError || 'Loading...'}
          </p>
        </div>
      )}

      {/* Query form */}
      <div className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <QueryForm
          onSubmit={handleQuery}
          isLoading={isLoading}
          disabled={!isReady}
        />
      </div>

      {/* Results */}
      <QueryResults result={result} error={error} isLoading={isLoading} />
    </div>
  );
}
