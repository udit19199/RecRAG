"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
		isUploading,
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
	const canChat =
		isReady &&
		!isIngesting &&
		(hasDocuments || ingestionStatus?.status === "complete");

	// Build extraction options based on vision config
	const getExtractionOptions = useCallback(():
		| ExtractionOptions
		| undefined => {
		if (!visionConfig) {
			return undefined;
		}
		return {
			extraction_mode: "vision_assisted",
			vision_provider: visionConfig.provider,
			vision_model: visionConfig.model,
		};
	}, [visionConfig]);

	// Wrap handleUpload to include extraction options
	const handleUploadWithVision = useCallback(
		async (files: File[]) => {
			const options = getExtractionOptions();
			await handleUpload(files, options);
		},
		[handleUpload, getExtractionOptions],
	);

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30">
			<div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
				<section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
					<div className="flex flex-1 flex-col overflow-hidden rounded-xl border bg-card">
						<div className="space-y-3 border-b bg-background px-4 py-3">
							{isIngesting ||
							ingestionStatus?.status === "complete" ||
							uploadFeedback?.type === "error" ? (
								<div className="animate-in fade-in slide-in-from-top-2 duration-300">
									<IngestionStatusDisplay
										status={ingestionStatus}
										error={statusError}
										isLoading={false}
									/>
									{ingestionStatus?.status === "complete" && !isIngesting ? (
										<div className="mt-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-600 dark:text-emerald-400">
											Processing complete — you can now ask questions.
										</div>
									) : null}
									{uploadFeedback?.type === "error" ? (
										<div className="mt-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
											{uploadFeedback.message}
										</div>
									) : null}
								</div>
							) : null}

							{uploadFeedback?.type === "success" && !isIngesting ? (
								<div className="animate-in fade-in slide-in-from-top-1 duration-300 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-xs text-emerald-600 dark:text-emerald-400">
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
