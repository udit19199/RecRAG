"use client";

import { IconFile, IconUpload, IconX } from "@tabler/icons-react";
import { type ChangeEvent, useRef, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import type { UploadOptions } from "@/lib/api/types";
import { cn } from "@/lib/utils";

interface FileUploaderProps {
	onUpload: (files: File[], options?: UploadOptions) => Promise<void>;
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
		await onUpload(selectedFiles, {
			extraction_mode: "text_only",
			replace: true,
		});
		setSelectedFiles([]);
	};

	const removeFile = (index: number) => {
		setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
	};

	return (
		<div className="flex flex-col gap-4">
			<label
				className={cn(
					"cursor-pointer text-center transition-colors",
					(disabled || isUploading) && "cursor-not-allowed opacity-50",
				)}
				onDrop={handleDrop}
				onDragOver={handleDragOver}
				onDragLeave={handleDragLeave}
			>
				<Card
					className={cn(
						"mx-auto max-w-lg border-dashed",
						isDragging && "border-ring bg-accent/40",
					)}
				>
					<CardContent className="flex flex-col items-center gap-2">
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
						<IconUpload className="text-muted-foreground" />
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
							This replaces your current document library
						</p>
					</CardContent>
				</Card>
			</label>

			{error ? (
				<Alert variant="destructive">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			) : null}

			{selectedFiles.length > 0 ? (
				<div className="flex flex-col gap-2">
					<h3 className="text-sm font-medium text-foreground">
						Selected files ({selectedFiles.length})
					</h3>
					<div className="flex flex-col gap-2">
						{selectedFiles.map((file, index) => (
							<Card key={file.name} size="sm">
								<CardContent className="flex items-center justify-between">
									<div className="flex items-center gap-3 overflow-hidden">
										<IconFile className="shrink-0 text-muted-foreground" />
										<span className="truncate text-sm text-foreground">
											{file.name}
										</span>
										<span className="text-xs text-muted-foreground">
											({(file.size / 1024 / 1024).toFixed(1)} MB)
										</span>
									</div>
									<Button
										type="button"
										variant="ghost"
										size="icon-xs"
										onClick={() => removeFile(index)}
										aria-label={`Remove ${file.name}`}
										disabled={isUploading}
									>
										<IconX data-icon="inline-start" />
									</Button>
								</CardContent>
							</Card>
						))}
					</div>
				</div>
			) : null}

			{selectedFiles.length > 0 ? (
				<Button
					type="button"
					onClick={handleUpload}
					disabled={isUploading || disabled || selectedFiles.length === 0}
					className="self-start"
				>
					{isUploading ? (
						<>
							<Spinner data-icon="inline-start" />
							Uploading…
						</>
					) : (
						`Replace corpus with ${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""}`
					)}
				</Button>
			) : null}
		</div>
	);
}
