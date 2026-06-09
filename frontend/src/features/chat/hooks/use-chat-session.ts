"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/features/chat/types";
import {
	checkRetrievalHealth,
	getEvalStatus,
	getIngestionStatus,
	getUploadedFiles,
	type IngestionStatus,
	queryRAG,
	uploadPDFs,
	waitForIngestionComplete,
} from "@/lib/api";
import type { ExtractionOptions } from "@/lib/api/types";

export function useChatSession() {
	const [isReady, setIsReady] = useState(false);
	const [hasDocuments, setHasDocuments] = useState(false);
	const [ingestionStatus, setIngestionStatus] =
		useState<IngestionStatus | null>(null);
	const [statusError, setStatusError] = useState<string | null>(null);
	const [healthError, setHealthError] = useState<string | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [uploadFeedback, setUploadFeedback] = useState<{
		type: "success" | "error";
		message: string;
	} | null>(null);
	const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [isQuerying, setIsQuerying] = useState(false);
	const messageIdRef = useRef(0);

	const nextId = () => {
		return messageIdRef.current++;
	};

	const fetchFiles = useCallback(async () => {
		try {
			const res = await getUploadedFiles();
			setUploadedFiles(res.files);
		} catch (_err) {
			// fail silently
		}
	}, []);

	const fetchIngestionStatus = useCallback(async () => {
		try {
			const nextStatus = await getIngestionStatus();
			setIngestionStatus(nextStatus);
			setStatusError(null);
			if (nextStatus.status === "complete") {
				void fetchFiles();
			}
		} catch (err) {
			setStatusError(
				err instanceof Error ? err.message : "Failed to fetch status",
			);
		}
	}, [fetchFiles]);

	useEffect(() => {
		const checkHealth = async () => {
			try {
				const health = await checkRetrievalHealth();
				setIsReady(health.pipeline_loaded ?? false);
				setHasDocuments(health.has_documents ?? false);

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
		fetchIngestionStatus();
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

	const handleUpload = async (files: File[], options?: ExtractionOptions) => {
		setIsUploading(true);
		setUploadFeedback(null);

		try {
			await uploadPDFs(files, options);
			setUploadFeedback({
				type: "success",
				message: `${files.length} file${files.length > 1 ? "s" : ""} uploaded. Existing corpus replaced and processing started.`,
			});
			const finalStatus = await waitForIngestionComplete();
			setIngestionStatus(finalStatus);
		} catch (err) {
			setUploadFeedback({
				type: "error",
				message: err instanceof Error ? err.message : "Upload failed",
			});
		} finally {
			setIsUploading(false);
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
		uploadFeedback,
		messages,
		isQuerying,
		handleQuery,
		handleUpload,
		handleReindexStarted,
	};
}
