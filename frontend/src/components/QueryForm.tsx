'use client';

import { useState, FormEvent } from 'react';

interface QueryFormProps {
  onSubmit: (query: string) => Promise<void>;
  isLoading: boolean;
  disabled?: boolean;
}

/**
 * Query form component for asking questions about documents
 *
 * Provides a text input for the query and a submit button.
 * Shows loading state while processing.
 */
export default function QueryForm({ onSubmit, isLoading, disabled }: QueryFormProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading || disabled) return;
    await onSubmit(query);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label
          htmlFor="query"
          className="text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Ask a question about your documents
        </label>
        <textarea
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., What is the main topic of these documents?"
          className="min-h-[100px] w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-zinc-900 placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:placeholder-zinc-500"
          disabled={disabled}
        />
      </div>

      <button
        type="submit"
        disabled={!query.trim() || isLoading || disabled}
        className="self-start rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <svg
              className="h-4 w-4 animate-spin"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Searching...
          </span>
        ) : (
          'Search'
        )}
      </button>
    </form>
  );
}
