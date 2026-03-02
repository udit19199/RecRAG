'use client';

import { useState, useEffect, useCallback } from 'react';
import FileUploader from '@/components/FileUploader';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
import { Separator } from '@/components/ui/separator';
import {
  uploadPDF,
  getIngestionStatus,
  waitForIngestionComplete,
  type IngestionStatus,
} from '@/lib/api';

export default function IngestPage() {
  const [status, setStatus] = useState<IngestionStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadFeedback, setUploadFeedback] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const newStatus = await getIngestionStatus();
      setStatus(newStatus);
      setStatusError(null);
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (status?.status !== 'processing') return;
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [status?.status, fetchStatus]);

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
      setStatus(finalStatus);
    } catch (err) {
      setUploadFeedback({
        type: 'error',
        message: err instanceof Error ? err.message : 'Upload failed',
      });
    } finally {
      setIsUploading(false);
    }
  };

  const isIngesting = status?.status === 'processing';

  return (
    <div className="mx-auto max-w-2xl px-6 py-10">
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-zinc-50">
          Documents
        </h1>
        <p className="mt-1 text-sm text-zinc-400">
          Upload PDF documents to be processed and indexed for querying.
        </p>
      </div>

      {/* Ingestion status */}
      <div className="mb-6">
        <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-400">
          Pipeline Status
        </h2>
        <IngestionStatusDisplay
          status={status}
          error={statusError}
          isLoading={false}
        />
      </div>

      <Separator className="mb-6" />

      {/* Upload feedback banner */}
      {uploadFeedback && (
        <div
            className={`mb-4 flex items-start gap-2 rounded-xl p-3 text-sm ${
              uploadFeedback.type === 'success'
                ? 'bg-green-950/40 text-green-300 soft-border'
                : 'bg-red-950 text-red-300 soft-border'
            }`}
        >
          {uploadFeedback.type === 'success' ? (
            <svg
              className="mt-0.5 h-4 w-4 shrink-0"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg
              className="mt-0.5 h-4 w-4 shrink-0"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
          )}
          <span>{uploadFeedback.message}</span>
        </div>
      )}

      {/* Upload card */}
      <div className="rounded-xl p-6 soft-border bg-zinc-900 shadow-sm">
        <h2 className="mb-4 text-sm font-semibold text-zinc-50">
          Upload PDFs
        </h2>
        <FileUploader
          onUpload={handleUpload}
          isUploading={isUploading}
          disabled={isIngesting}
        />
      </div>

      {/* Tips */}
      <div className="mt-5 rounded-xl p-4 soft-border bg-zinc-900/60">
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
    </div>
  );
}
