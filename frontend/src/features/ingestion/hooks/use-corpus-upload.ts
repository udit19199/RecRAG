"use client";

import { useEffect, useState } from "react";
import {
	getIngestionStatus,
	getUploadedFiles,
	uploadPDFs,
	waitForIngestionComplete,
} from "@/lib/api";
import type { IngestionStatus, UploadOptions } from "@/lib/api/types";

type UploadOutcome =
	| { ok: true; fileCount: number }
	| { ok: false; error: string };

async function performUpload(
	files: File[],
	options: UploadOptions | undefined,
	onProgress: (percent: number) => void,
): Promise<UploadOutcome> {
	try {
		await uploadPDFs(files, options, onProgress);
		return { ok: true, fileCount: files.length };
	} catch (err) {
		return {
			ok: false,
			error: err instanceof Error ? err.message : "Upload failed",
		};
	}
}

async function loadUploadedFiles(): Promise<string[]> {
	try {
		const res = await getUploadedFiles();
		return res.files;
	} catch {
		return [];
	}
}

async function loadIngestionStatus(): Promise<IngestionStatus | null> {
	try {
		return await getIngestionStatus();
	} catch {
		return null;
	}
}

export function useCorpusUpload() {
	const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
	const [ingestionStatus, setIngestionStatus] =
		useState<IngestionStatus | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [uploadProgress, setUploadProgress] = useState<number | null>(null);
	const [feedback, setFeedback] = useState<{
		type: "success" | "error";
		message: string;
	} | null>(null);

	const refresh = async () => {
		const [files, status] = await Promise.all([
			loadUploadedFiles(),
			loadIngestionStatus(),
		]);
		setUploadedFiles(files);
		setIngestionStatus(status);
	};

	useEffect(() => {
		let cancelled = false;
		const load = async () => {
			const [files, status] = await Promise.all([
				loadUploadedFiles(),
				loadIngestionStatus(),
			]);
			if (cancelled) return;
			setUploadedFiles(files);
			setIngestionStatus(status);
		};
		void load();
		return () => {
			cancelled = true;
		};
	}, []);

	useEffect(() => {
		if (ingestionStatus?.status !== "processing") return;
		const interval = window.setInterval(() => {
			void loadIngestionStatus().then(setIngestionStatus);
		}, 3000);
		return () => window.clearInterval(interval);
	}, [ingestionStatus?.status]);

	const upload = async (files: File[], options?: UploadOptions) => {
		setIsUploading(true);
		setUploadProgress(0);
		setFeedback(null);

		const result = await performUpload(files, options, setUploadProgress);
		if (!result.ok) {
			setIsUploading(false);
			setUploadProgress(null);
			setFeedback({ type: "error", message: result.error });
			return;
		}

		setUploadProgress(null);
		setFeedback({
			type: "success",
			message: `${result.fileCount} file${result.fileCount > 1 ? "s" : ""} uploaded. Indexing…`,
		});

		const finalStatus = await waitForIngestionComplete(
			2000,
			120000,
			setIngestionStatus,
		);
		setIngestionStatus(finalStatus);
		const filesOnDisk = await loadUploadedFiles();
		setUploadedFiles(filesOnDisk);
		setIsUploading(false);
		setUploadProgress(null);
	};

	return {
		uploadedFiles,
		ingestionStatus,
		isUploading,
		uploadProgress,
		feedback,
		upload,
		refresh,
	};
}
