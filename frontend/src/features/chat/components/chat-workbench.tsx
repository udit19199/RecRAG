"use client";

import { useEffect, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardFooter,
	CardHeader,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ChatInput } from "@/features/chat/components/chat-input";
import { ChatMessageView } from "@/features/chat/components/chat-message";
import { EmptyState } from "@/features/chat/components/empty-state";
import { TypingIndicator } from "@/features/chat/components/typing-indicator";
import { useChatSession } from "@/features/chat/hooks/use-chat-session";
import IngestionStatusDisplay from "@/features/ingestion/components/ingestion-status";
import ModelPicker, {
	type VisionConfig,
} from "@/features/model-compare/components/model-picker";
import { readStoredVisionConfig } from "@/features/model-compare/lib";
import type { ExtractionOptions, UploadOptions } from "@/lib/api/types";

export function ChatWorkbench() {
	const {
		isReady,
		hasDocuments,
		uploadedFiles,
		ingestionStatus,
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
	} = useChatSession();

	const chatBottomRef = useRef<HTMLDivElement>(null);
	const [visionConfig, setVisionConfig] = useState<VisionConfig | null>(
		readStoredVisionConfig,
	);

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const isIngesting = ingestionStatus?.status === "processing" || isReindexing;
	const needsIndexing =
		uploadedFiles.length > 0 &&
		!hasDocuments &&
		!isIngesting &&
		ingestionStatus?.status !== "complete";
	const isAwaitingIngestion =
		isUploading &&
		uploadProgress === null &&
		uploadFeedback?.type === "success";
	const canChat =
		isReady &&
		!isIngesting &&
		(hasDocuments || ingestionStatus?.status === "complete");

	const getExtractionOptions = (): ExtractionOptions | undefined => {
		if (!visionConfig) {
			return undefined;
		}
		return {
			extraction_mode: "vision_assisted",
			vision_provider: visionConfig.provider,
			vision_model: visionConfig.model,
		};
	};

	const handleReindexWithVision = async () => {
		await handleReindex(getExtractionOptions());
	};

	const handleUploadWithVision = async (
		files: File[],
		uploadOpts?: Pick<UploadOptions, "replace">,
	) => {
		const extraction = getExtractionOptions();
		if (!extraction && !uploadOpts?.replace) {
			await handleUpload(files);
			return;
		}

		await handleUpload(files, {
			...(extraction ?? { extraction_mode: "text_only" as const }),
			...(uploadOpts?.replace ? { replace: true } : {}),
		});
	};

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30">
			<div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
				<section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
					<Card className="flex flex-1 flex-col overflow-hidden py-0">
						<CardHeader className="flex flex-col gap-3 border-b bg-background">
							{isUploading && uploadProgress !== null ? (
								<Card
									size="sm"
									className="animate-in bg-muted/40 duration-300 fade-in slide-in-from-top-2"
								>
									<CardContent>
										<div className="flex items-center justify-between gap-3 text-sm">
											<span className="font-medium text-foreground">
												Uploading files…
											</span>
											<span className="text-xs tabular-nums text-muted-foreground">
												{uploadProgress}%
											</span>
										</div>
										<Progress
											className="mt-3"
											value={uploadProgress}
											aria-label="Upload progress"
										/>
									</CardContent>
								</Card>
							) : null}

							{needsIndexing ? (
								<Alert className="animate-in duration-300 fade-in slide-in-from-top-2">
									<AlertTitle>Documents uploaded but not indexed</AlertTitle>
									<AlertDescription className="flex flex-wrap items-center justify-between gap-3">
										<span>
											{uploadedFiles.length} file
											{uploadedFiles.length === 1 ? "" : "s"} on disk, but none
											are searchable yet. Re-index to enable chat.
										</span>
										<Button
											type="button"
											size="sm"
											onClick={handleReindexWithVision}
											disabled={isReindexing || isUploading}
										>
											{isReindexing ? "Re-indexing…" : "Re-index documents"}
										</Button>
									</AlertDescription>
								</Alert>
							) : null}

							{isIngesting ||
							isAwaitingIngestion ||
							ingestionStatus?.status === "complete" ||
							ingestionStatus?.status === "error" ||
							uploadFeedback?.type === "error" ? (
								<div className="animate-in duration-300 fade-in slide-in-from-top-2">
									<IngestionStatusDisplay
										status={ingestionStatus}
										error={statusError}
										isLoading={isAwaitingIngestion && !isIngesting}
									/>
									{ingestionStatus?.status === "complete" && !isIngesting ? (
										<Alert className="mt-2">
											<AlertDescription>
												Processing complete, you can now ask questions.
											</AlertDescription>
										</Alert>
									) : null}
									{uploadFeedback?.type === "error" ? (
										<Alert variant="destructive" className="mt-2">
											<AlertDescription>
												{uploadFeedback.message}
											</AlertDescription>
										</Alert>
									) : null}
								</div>
							) : null}

							{uploadFeedback?.type === "success" && isIngesting ? (
								<Alert className="animate-in duration-300 fade-in slide-in-from-top-1">
									<AlertDescription>{uploadFeedback.message}</AlertDescription>
								</Alert>
							) : null}

							<div className="flex flex-wrap items-center gap-3">
								<ModelPicker
									disabled={isIngesting}
									onReindexStarted={handleReindexStarted}
									onConfigChanged={() => {}}
									onVisionConfigChanged={setVisionConfig}
								/>
							</div>

							{uploadedFiles && uploadedFiles.length > 0 ? (
								<div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
									<span className="font-medium">Active Files:</span>
									{uploadedFiles.map((file) => (
										<Badge key={file} variant="secondary">
											{file}
										</Badge>
									))}
								</div>
							) : null}
						</CardHeader>

						<CardContent className="h-full overflow-y-auto p-4 md:p-6">
							{messages.length === 0 ? (
								<EmptyState
									isReady={isReady}
									error={healthError}
									onUpload={handleUploadWithVision}
									isUploading={isUploading}
								/>
							) : (
								<div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
									{messages.map((message) => (
										<ChatMessageView key={message.id} message={message} />
									))}
									{isQuerying ? <TypingIndicator /> : null}
									<div ref={chatBottomRef} />
								</div>
							)}
						</CardContent>
					</Card>

					<Card className="shrink-0">
						<CardFooter className="border-0 bg-transparent p-4">
							<ChatInput
								onSubmit={handleQuery}
								onUpload={handleUploadWithVision}
								isLoading={isQuerying}
								isUploading={isUploading || isIngesting}
								disabled={!canChat || isIngesting}
								disabledReason={
									isIngesting
										? "Document indexing in progress…"
										: !isReady
											? "Waiting for retrieval pipeline…"
											: ingestionStatus?.status === "error" || needsIndexing
												? "Documents need indexing. Use Re-index to enable chat."
												: !(
															hasDocuments ||
															ingestionStatus?.status === "complete"
														)
													? "Upload and index your documents before chatting."
													: undefined
								}
							/>
						</CardFooter>
					</Card>
				</section>
			</div>
		</div>
	);
}
