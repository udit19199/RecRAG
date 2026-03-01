'use client';

import { useState, useEffect, useCallback } from 'react';
import FileUploader from '@/components/FileUploader';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
import QueryForm from '@/components/QueryForm';
import QueryResults from '@/components/QueryResults';
import {
  queryRAG,
  checkRetrievalHealth,
  uploadPDF,
  getIngestionStatus,
  waitForIngestionComplete,
  type QueryResponse,
  type IngestionStatus,
} from '@/lib/api';

export default function HomePage() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fetchIngestionStatus = useCallback(async () => {
    try {
      const newStatus = await getIngestionStatus();
      setIngestionStatus(newStatus);
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
          setHealthError('Retrieval pipeline is still loading. Please wait a moment.');
        }
      } catch (err) {
        setHealthError('Unable to connect to retrieval API. Please ensure the service is running.');
      }
    };

    checkHealth();
    fetchIngestionStatus();
  }, [fetchIngestionStatus]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (ingestionStatus?.status === 'processing') {
        fetchIngestionStatus();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [ingestionStatus?.status, fetchIngestionStatus]);

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

  const handleUpload = async (files: File[]) => {
    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      for (const file of files) {
        await uploadPDF(file);
      }

      setUploadSuccess(
        `Successfully uploaded ${files.length} file${files.length > 1 ? 's' : ''}. Processing has started.`
      );

      const finalStatus = await waitForIngestionComplete();
      setIngestionStatus(finalStatus);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          RecRAG
        </h1>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">
          Upload documents and ask questions in a unified chat interface.
        </p>
      </div>

      {!isReady && !isLoading && (
        <div className="mb-6 flex items-center gap-3 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-900 dark:bg-yellow-950/50">
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
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            {healthError || 'Loading...'}
          </p>
        </div>
      )}

      <div className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Upload Documents
        </h2>
        
        <div className="mb-4">
          <IngestionStatusDisplay
            status={ingestionStatus}
            error={statusError}
            isLoading={false}
          />
        </div>

        {uploadSuccess && (
          <div className="mb-4 flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950/50">
            <svg
              className="h-5 w-5 text-green-500"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-green-700 dark:text-green-300">{uploadSuccess}</p>
          </div>
        )}

        {uploadError && (
          <div className="mb-4 flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950/50">
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
            <p className="text-sm text-red-700 dark:text-red-300">{uploadError}</p>
          </div>
        )}

        <FileUploader
          onUpload={handleUpload}
          isUploading={isUploading}
          disabled={ingestionStatus?.status === 'processing'}
        />
      </div>

      <div className="mb-8 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <QueryForm
          onSubmit={handleQuery}
          isLoading={isLoading}
          disabled={!isReady}
        />
      </div>

      <QueryResults result={result} error={error} isLoading={isLoading} />
    </div>
  );
}
