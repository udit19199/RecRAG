'use client';

import { useState, useEffect, useCallback } from 'react';
import FileUploader from '@/components/FileUploader';
import IngestionStatusDisplay from '@/components/IngestionStatusDisplay';
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
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Poll status periodically
  const fetchStatus = useCallback(async () => {
    try {
      const newStatus = await getIngestionStatus();
      setStatus(newStatus);
      setStatusError(null);
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchStatus();

    // Poll every 3 seconds when there's activity
    const interval = setInterval(() => {
      if (status?.status === 'processing') {
        fetchStatus();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchStatus, status?.status]);

  const handleUpload = async (files: File[]) => {
    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      // Upload each file
      for (const file of files) {
        await uploadPDF(file);
      }

      setUploadSuccess(
        `Successfully uploaded ${files.length} file${files.length > 1 ? 's' : ''}. Processing has started.`
      );

      // Wait for processing to complete
      const finalStatus = await waitForIngestionComplete();
      setStatus(finalStatus);
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
          Ingest Documents
        </h1>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">
          Upload PDF documents to be processed and indexed for querying.
          The ingestion pipeline will automatically chunk and embed the content.
        </p>
      </div>

      {/* Status display */}
      <div className="mb-8">
        <h2 className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          Ingestion Status
        </h2>
        <IngestionStatusDisplay
          status={status}
          error={statusError}
          isLoading={false}
        />
      </div>

      {/* Upload success message */}
      {uploadSuccess && (
        <div className="flex items-center gap-3 py-4">
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
          <p className="text-green-600 dark:text-green-400">{uploadSuccess}</p>
        </div>
      )}

      {/* Upload error message */}
      {uploadError && (
        <div className="flex items-center gap-3 py-4">
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
          <p className="text-red-600 dark:text-red-400">{uploadError}</p>
        </div>
      )}

      {/* File uploader */}
      <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <h2 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Upload PDFs
        </h2>
        <FileUploader
          onUpload={handleUpload}
          isUploading={isUploading}
          disabled={status?.status === 'processing'}
        />
      </div>

      {/* Help text */}
      <div className="mt-6 rounded-lg bg-zinc-100 p-4 dark:bg-zinc-900">
        <h3 className="mb-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          Tips for better results
        </h3>
        <ul className="space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
          <li>• Use well-formatted PDFs with clear text (not scanned images)</li>
          <li>• Keep file size under 50MB for optimal processing</li>
          <li>• After upload, wait for status to show &quot;Complete&quot; before querying</li>
          <li>• Multiple files can be uploaded at once</li>
        </ul>
      </div>
    </div>
  );
}
