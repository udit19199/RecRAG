"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/features/chat/types";
import {
	checkIndexStatus,
	checkRetrievalHealth,
	getEvalStatus,
	getIngestionStatus,
	getUploadedFiles,
	type IngestionStatus,
	queryRAG,
	reloadRetrievalConfig,
	triggerReindex,
	uploadPDFs,
	waitForIngestionComplete,
} from "@/lib/api";
import type { ExtractionOptions, UploadOptions } from "@/lib/api/types";

export function useChatSession() {
	const [isReady, setIsReady] = useState(false);
	const [hasDocuments, setHasDocuments] = useState(false);
	const [ingestionStatus, setIngestionStatus] =
		useState<IngestionStatus | null>(null);
	const [statusError, setStatusError] = useState<string | null>(null);
	const [healthError, setHealthError] = useState<string | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [uploadProgress, setUploadProgress] = useState<number | null>(null);
	const [uploadFeedback, setUploadFeedback] = useState<{
		type: "success" | "error";
		message: string;
	} | null>(null);
	const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [isQuerying, setIsQuerying] = useState(false);
	const [isReindexing, setIsReindexing] = useState(false);
	const messageIdRef = useRef(0);

	const nextId = () => {
		return messageIdRef.current++;
	};

	const fetchFiles = useCallback(async () => {
		try {
			const res = await getUploadedFiles();
			setUploadedFiles(res.files);
			// #region agent log
			fetch(
				"http://127.0.0.1:7916/ingest/3188a69e-7db0-4d0f-b5ef-4e3827cb1095",
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"X-Debug-Session-Id": "50063e",
					},
					body: JSON.stringify({
						sessionId: "50063e",
						location: "use-chat-session.ts:files",
						message: "uploaded files fetched",
						data: { fileCount: res.files.length, files: res.files },
						timestamp: Date.now(),
						hypothesisId: "H4",
					}),
				},
			).catch(() => {});
			// #endregion
		} catch (_err) {
			// fail silently
		}
	}, []);

	const refreshHealth = useCallback(async () => {
		try {
			const health = await checkRetrievalHealth();
			let hasDocs = health.has_documents ?? false;
			if (!hasDocs) {
				try {
					const indexStatus = await checkIndexStatus({});
					hasDocs = indexStatus.has_documents;
				} catch {
					// Ignore index status errors; rely on retrieval health.
				}
			}
			setIsReady(health.pipeline_loaded ?? false);
			setHasDocuments(hasDocs);

			if (!health.pipeline_loaded && health.error_message) {
				setHealthError(health.error_message);
			} else {
				setHealthError(null);
			}
		} catch (err) {
			setIsReady(false);
			setHealthError(
				err instanceof Error ? err.message : "Service unreachable",
			);
		}
	}, []);

	const syncRetrievalAfterIngestion = useCallback(async () => {
		try {
			const reloaded = await reloadRetrievalConfig();
			// #region agent log
			fetch(
				"http://127.0.0.1:7916/ingest/3188a69e-7db0-4d0f-b5ef-4e3827cb1095",
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"X-Debug-Session-Id": "50063e",
					},
					body: JSON.stringify({
						sessionId: "50063e",
						runId: "post-fix",
						location: "use-chat-session.ts:reload",
						message: "retrieval pipeline reloaded from config.toml",
						data: {
							embedding: reloaded.embedding,
							has_documents_pending: true,
						},
						timestamp: Date.now(),
						hypothesisId: "H6",
					}),
				},
			).catch(() => {});
			// #endregion
		} catch {
			// Non-fatal: chat may still work via ingestion status gate.
		}
		await refreshHealth();
	}, [refreshHealth]);

	const fetchIngestionStatus = useCallback(async () => {
		try {
			const nextStatus = await getIngestionStatus();
			setIngestionStatus(nextStatus);
			setStatusError(null);
			// #region agent log
			fetch(
				"http://127.0.0.1:7916/ingest/3188a69e-7db0-4d0f-b5ef-4e3827cb1095",
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"X-Debug-Session-Id": "50063e",
					},
					body: JSON.stringify({
						sessionId: "50063e",
						location: "use-chat-session.ts:ingestion-status",
						message: "ingestion status fetched",
						data: {
							status: nextStatus.status,
							error_message: nextStatus.error_message,
							files_processed: nextStatus.files_processed,
						},
						timestamp: Date.now(),
						hypothesisId: "H2",
					}),
				},
			).catch(() => {});
			// #endregion
			if (nextStatus.status === "complete") {
				void fetchFiles();
				void syncRetrievalAfterIngestion();
			}
		} catch (err) {
			setStatusError(
				err instanceof Error ? err.message : "Failed to fetch status",
			);
		}
	}, [fetchFiles, syncRetrievalAfterIngestion]);

	useEffect(() => {
		const checkHealth = async () => {
			try {
				const health = await checkRetrievalHealth();
				setIsReady(health.pipeline_loaded ?? false);
				setHasDocuments(health.has_documents ?? false);
				// #region agent log
				fetch(
					"http://127.0.0.1:7916/ingest/3188a69e-7db0-4d0f-b5ef-4e3827cb1095",
					{
						method: "POST",
						headers: {
							"Content-Type": "application/json",
							"X-Debug-Session-Id": "50063e",
						},
						body: JSON.stringify({
							sessionId: "50063e",
							runId: "post-fix",
							location: "use-chat-session.ts:health",
							message: "retrieval health checked",
							data: {
								pipeline_loaded: health.pipeline_loaded,
								has_documents: health.has_documents,
							},
							timestamp: Date.now(),
							hypothesisId: "H1",
						}),
					},
				).catch(() => {});
				// #endregion

				if (!health.pipeline_loaded && health.error_message) {
					setHealthError(health.error_message);
				} else {
					setHealthError(null);
				}
			} catch (err) {
				setIsReady(false);
				setHealthError(
					err instanceof Error ? err.message : "Service unreachable",
				);
			}
		};

		checkHealth();
		fetchIngestionStatus();
		fetchFiles();
	}, [fetchIngestionStatus, fetchFiles]);

	useEffect(() => {
		if (ingestionStatus?.status !== "processing") return;

		const interval = setInterval(fetchIngestionStatus, 3000);
		return () => clearInterval(interval);
	}, [fetchIngestionStatus, ingestionStatus?.status]);

	const handleReindexStarted = () => {
		void fetchIngestionStatus();
	};

	const handleReindex = async (options?: ExtractionOptions) => {
		setIsReindexing(true);
		setUploadFeedback(null);

		try {
			await triggerReindex(options);
			void fetchIngestionStatus();
			const finalStatus = await waitForIngestionComplete(
				2000,
				120000,
				setIngestionStatus,
			);
			setIngestionStatus(finalStatus);
			await fetchFiles();
			await syncRetrievalAfterIngestion();
		} catch (err) {
			setUploadFeedback({
				type: "error",
				message: err instanceof Error ? err.message : "Re-index failed",
			});
		} finally {
			setIsReindexing(false);
		}
	};

	const handleQuery = async (query: string) => {
		if (!query.trim() || isQuerying) return;

		setMessages((prev) => [
			...prev,
			{ id: nextId(), role: "user", content: query },
		]);
		setIsQuerying(true);

		try {
			const result = await queryRAG({ query });
			const assistantId = nextId();
			setMessages((prev) => [
				...prev,
				{
					id: assistantId,
					role: "assistant",
					content: result.response,
					context: result.context,
					...(result.eval_job_id
						? { eval_job_id: result.eval_job_id, eval_status: "pending" }
						: {}),
				},
			]);

			if (result.eval_job_id) {
				const jobId = result.eval_job_id;
				void (async () => {
					try {
						for (;;) {
							const status = await getEvalStatus(jobId);
							if (status.status === "complete") {
								setMessages((prev) =>
									prev.map((message) =>
										message.role === "assistant" &&
										message.eval_job_id === jobId
											? {
													...message,
													...(status.scores ? { eval: status.scores } : {}),
													eval_status: "complete",
												}
											: message,
									),
								);
								break;
							}

							if (status.status === "error") {
								setMessages((prev) =>
									prev.map((message) =>
										message.role === "assistant" &&
										message.eval_job_id === jobId
											? { ...message, eval_status: "error" }
											: message,
									),
								);
								break;
							}

							await new Promise((resolve) => setTimeout(resolve, 1500));
						}
					} catch {
						// Ignore polling errors.
					}
				})();
			}
		} catch (err) {
			setMessages((prev) => [
				...prev,
				{
					id: nextId(),
					role: "assistant",
					content: "",
					error:
						err instanceof Error ? err.message : "An unexpected error occurred",
				},
			]);
		} finally {
			setIsQuerying(false);
		}
	};

	const handleUpload = async (files: File[], options?: UploadOptions) => {
		setIsUploading(true);
		setUploadProgress(0);
		setUploadFeedback(null);

		try {
			await uploadPDFs(files, options, setUploadProgress);
			setUploadProgress(null);
			setUploadFeedback({
				type: "success",
				message: `${files.length} file${files.length > 1 ? "s" : ""} uploaded. Indexing documents…`,
			});
			const finalStatus = await waitForIngestionComplete(
				2000,
				120000,
				setIngestionStatus,
			);
			setIngestionStatus(finalStatus);
			await fetchFiles();
			await syncRetrievalAfterIngestion();
		} catch (err) {
			setUploadFeedback({
				type: "error",
				message: err instanceof Error ? err.message : "Upload failed",
			});
		} finally {
			setIsUploading(false);
			setUploadProgress(null);
		}
	};

	return {
		isReady,
		hasDocuments,
		ingestionStatus,
		uploadedFiles,
		statusError,
		healthError,
		isUploading,
		uploadProgress,
		uploadFeedback,
		messages,
		isQuerying,
		isReindexing,
		handleQuery,
		handleUpload,
		handleReindex,
		handleReindexStarted,
	};
}
