"use client";

import { useEffect, useRef, useState } from "react";
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
import type { ExtractionOptions } from "@/lib/api/types";

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
		handleQuery,
		handleUpload,
		handleReindexStarted,
	} = useChatSession();

	const chatBottomRef = useRef<HTMLDivElement>(null);
	const [visionConfig, setVisionConfig] = useState<VisionConfig | null>(null);

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const isIngesting = ingestionStatus?.status === "processing";
	const isAwaitingIngestion =
		isUploading &&
		uploadProgress === null &&
		uploadFeedback?.type === "success";
	const canChat =
		isReady &&
		!isIngesting &&
		(hasDocuments || ingestionStatus?.status === "complete");

	// Build extraction options based on vision config
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

	// Wrap handleUpload to include extraction options
	const handleUploadWithVision = async (files: File[]) => {
		const options = getExtractionOptions();
		await handleUpload(files, options);
	};

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30">
			<div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
				<section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
					<div className="flex flex-1 flex-col overflow-hidden rounded-xl border bg-card">
						<div className="space-y-3 border-b bg-background px-4 py-3">
							{isUploading && uploadProgress !== null ? (
								<div className="animate-in fade-in slide-in-from-top-2 duration-300 rounded-lg border border-border bg-muted/40 px-4 py-3">
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
								</div>
							) : null}

							{isIngesting ||
							isAwaitingIngestion ||
							ingestionStatus?.status === "complete" ||
							uploadFeedback?.type === "error" ? (
								<div className="animate-in fade-in slide-in-from-top-2 duration-300">
									<IngestionStatusDisplay
										status={ingestionStatus}
										error={statusError}
										isLoading={isAwaitingIngestion && !isIngesting}
									/>
									{ingestionStatus?.status === "complete" && !isIngesting ? (
										<div className="mt-2 rounded-lg border border-success/20 bg-success/5 px-3 py-2 text-xs text-success dark:text-success-foreground">
											Processing complete, you can now ask questions.
										</div>
									) : null}
									{uploadFeedback?.type === "error" ? (
										<div className="mt-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
											{uploadFeedback.message}
										</div>
									) : null}
								</div>
							) : null}

							{uploadFeedback?.type === "success" && isIngesting ? (
								<div className="animate-in fade-in slide-in-from-top-1 duration-300 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs text-foreground">
									{uploadFeedback.message}
								</div>
							) : null}

							<div className="flex flex-wrap items-center gap-3">
								<ModelPicker
									disabled={isIngesting}
									onReindexStarted={handleReindexStarted}
									onConfigChanged={() => {}}
									onVisionConfigChanged={setVisionConfig}
								/>
							</div>

							{uploadedFiles && uploadedFiles.length > 0 && (
								<div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
									<span className="font-medium">Active Files:</span>
									{uploadedFiles.map((file) => (
										<span
											key={file}
											className="rounded-md bg-secondary px-2 py-0.5 border border-border"
										>
											{file}
										</span>
									))}
								</div>
							)}
						</div>

						<div className="h-full overflow-y-auto p-4 md:p-6">
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
						</div>
					</div>

					<div className="shrink-0 rounded-xl border bg-card p-4">
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
										: !(hasDocuments || ingestionStatus?.status === "complete")
											? "Upload and index your documents before chatting."
											: undefined
							}
						/>
					</div>
				</section>
			</div>
		</div>
	);
}
