"use client";

import { type ChangeEvent, useRef, useState } from "react";

interface FileUploaderProps {
	onUpload: (files: File[]) => Promise<void>;
	isUploading: boolean;
	disabled?: boolean;
}

export default function FileUploader({
	onUpload,
	isUploading,
	disabled,
}: FileUploaderProps) {
	const [isDragging, setIsDragging] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
	const fileInputRef = useRef<HTMLInputElement>(null);

	const validateFiles = (files: FileList | null): File[] => {
		if (!files) return [];

		const validFiles: File[] = [];
		const errors: string[] = [];

		for (const file of Array.from(files)) {
			if (!file.name.toLowerCase().endsWith(".pdf")) {
				errors.push(`${file.name} - Only PDF files are allowed`);
			} else if (file.size > 50 * 1024 * 1024) {
				errors.push(`${file.name} - File exceeds 50MB limit`);
			} else {
				validFiles.push(file);
			}
		}

		setError(errors.length > 0 ? errors.join("\n") : null);
		return validFiles;
	};

	const handleFiles = (files: FileList | null) => {
		const validFiles = validateFiles(files);
		if (validFiles.length > 0) setSelectedFiles(validFiles);
	};

	const handleDrop = (event: React.DragEvent) => {
		event.preventDefault();
		setIsDragging(false);
		if (disabled || isUploading) return;
		handleFiles(event.dataTransfer.files);
	};

	const handleDragOver = (event: React.DragEvent) => {
		event.preventDefault();
		setIsDragging(true);
	};

	const handleDragLeave = () => setIsDragging(false);

	const onFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
		handleFiles(event.target.files);
	};

	const handleUpload = async () => {
		if (selectedFiles.length === 0 || isUploading || disabled) return;
		await onUpload(selectedFiles);
		setSelectedFiles([]);
	};

	const removeFile = (index: number) => {
		setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
	};

	return (
		<div className="flex flex-col gap-4">
			<label
				className={`cursor-pointer text-center transition-colors ${disabled || isUploading ? "cursor-not-allowed opacity-50" : ""}`}
				onDrop={handleDrop}
				onDragOver={handleDragOver}
				onDragLeave={handleDragLeave}
			>
				<div
					className={`mx-auto max-w-lg rounded-lg border border-dashed border-input bg-card p-4 ${isDragging ? "border-ring bg-accent/40" : ""}`}
				>
					<input
						ref={fileInputRef}
						type="file"
						accept=".pdf"
						multiple
						onChange={onFileInputChange}
						aria-label="Select PDF files to upload"
						className="hidden"
						disabled={disabled || isUploading}
					/>

					<div className="flex flex-col items-center gap-2">
						<svg
							aria-hidden="true"
							className="size-7 text-muted-foreground"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							strokeWidth="1.5"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								d="M9 8.25H7.5a2.25 2.25 0 00-2.25 2.25v9a2.25 2.25 0 002.25 2.25h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25H15m0-3l-3-3m0 0l-3 3m3-3v11.25"
							/>
						</svg>
						<div className="text-foreground">
							<span className="font-medium text-foreground">
								Upload your full batch
							</span>{" "}
							or drag and drop
						</div>
						<p className="text-xs text-muted-foreground">
							PDF files only (max 50MB)
						</p>
						<p className="text-xs text-muted-foreground">
							A new batch replaces the current corpus
						</p>
					</div>
				</div>
			</label>

			{error && (
				<div
					className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive"
					role="alert"
				>
					{error}
				</div>
			)}

			{selectedFiles.length > 0 && (
				<div className="flex flex-col gap-2">
					<h3 className="text-sm font-medium text-foreground">
						Selected files ({selectedFiles.length})
					</h3>
					<div className="flex flex-col gap-2">
						{selectedFiles.map((file, index) => (
							<div
								key={file.name}
								className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-2"
							>
								<div className="flex items-center gap-3 overflow-hidden">
									<svg
										aria-hidden="true"
										className="size-5 flex-shrink-0 text-muted-foreground"
										viewBox="0 0 24 24"
										fill="currentColor"
									>
										<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
										<path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
									</svg>
									<span className="truncate text-sm text-foreground">
										{file.name}
									</span>
									<span className="text-xs text-muted-foreground">
										({(file.size / 1024 / 1024).toFixed(1)} MB)
									</span>
								</div>
								<button
									type="button"
									onClick={() => removeFile(index)}
									aria-label={`Remove ${file.name}`}
									className="flex-shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
									disabled={isUploading}
								>
									<svg
										aria-hidden="true"
										className="size-4"
										viewBox="0 0 20 20"
										fill="currentColor"
									>
										<path
											fillRule="evenodd"
											d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
											clipRule="evenodd"
										/>
									</svg>
								</button>
							</div>
						))}
					</div>
				</div>
			)}

			{selectedFiles.length > 0 && (
				<button
					type="button"
					onClick={handleUpload}
					disabled={isUploading || disabled || selectedFiles.length === 0}
					className="self-start rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
				>
					{isUploading ? (
						<span className="flex items-center gap-2">
							<svg
								aria-hidden="true"
								className="size-4 animate-spin"
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
							Uploading…
						</span>
					) : (
						`Replace corpus with ${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""}`
					)}
				</button>
			)}
		</div>
	);
}
