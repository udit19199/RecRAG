'use client';

import type { IngestionStatus } from '@/lib/api';

interface IngestionStatusDisplayProps {
  status: IngestionStatus | null;
  error: string | null;
  isLoading: boolean;
}

/**
 * Ingestion status display component
 *
 * Shows the current status of the ingestion process with appropriate
 * visual indicators and details.
 */
export default function IngestionStatusDisplay({
  status,
  error,
  isLoading,
}: IngestionStatusDisplayProps) {
  if (isLoading && !status) {
    return (
      <div className="flex items-center gap-2 text-zinc-500">
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
        <span className="text-sm">Loading status...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2">
        <svg
          className="h-5 w-5 text-red-500"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
        <p className="text-sm text-red-400">{error}</p>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  const statusConfig = {
    idle: {
      label: 'Idle',
      description: 'No ingestion in progress. Upload files to start processing.',
      color: 'text-zinc-400',
      icon: (
        <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
            clipRule="evenodd"
          />
        </svg>
      ),
    },
    processing: {
      label: 'Processing',
      description: `Started at ${status.started_at || 'unknown'}. Waiting for ingestion to complete...`,
      color: 'text-blue-400',
      icon: (
        <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
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
      ),
    },
    complete: {
      label: 'Complete',
      description: `Successfully processed ${status.files_processed || 0} files.`,
      color: 'text-green-400',
      icon: (
        <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
      ),
    },
    error: {
      label: 'Error',
      description: status.error_message || 'An unknown error occurred.',
      color: 'text-red-400',
      icon: (
        <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      ),
    },
  };

  const config = statusConfig[status.status];

  return (
    <div className="flex items-center gap-3">
      <div className={`flex-shrink-0 ${config.color}`}>{config.icon}</div>
      <div>
        <p className={`text-sm font-medium ${config.color}`}>
          {config.label}: {config.description}
        </p>
        {status.completed_at && (
          <p className="text-xs text-zinc-400 mt-1">
            Completed at: {status.completed_at}
          </p>
        )}
      </div>
    </div>
  );
}
