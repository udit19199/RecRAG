"use client";

import { useEffect, useRef } from "react";
import { ChatInput } from "@/features/chat/components/chat-input";
import { ChatMessageView } from "@/features/chat/components/chat-message";
import { EmptyState } from "@/features/chat/components/empty-state";
import { TypingIndicator } from "@/features/chat/components/typing-indicator";
import { useChatSession } from "@/features/chat/hooks/use-chat-session";
import IngestionStatusDisplay from "@/features/ingestion/components/ingestion-status";
import ModelPicker from "@/features/model-compare/components/model-picker";

export function ChatWorkbench() {
	const {
		isReady,
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

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const isIngesting = ingestionStatus?.status === "processing";
	const canChat =
		isReady && ingestionStatus?.status === "complete" && !isIngesting;

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30">
			<div className="mx-auto flex min-h-0 w-full max-w-[1440px] flex-1 gap-4 overflow-hidden p-4">
				<section className="flex min-w-0 flex-1 flex-col gap-4 overflow-hidden">
					<div className="flex flex-1 flex-col overflow-hidden rounded-xl border bg-card">
						<div className="space-y-3 border-b bg-background px-4 py-3">
							{isIngesting || uploadFeedback?.type === "error" ? (
								<div className="animate-in fade-in slide-in-from-top-2 duration-300">
									<IngestionStatusDisplay
										status={ingestionStatus}
										error={statusError}
										isLoading={false}
									/>
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
								/>
							</div>
						</div>

						<div className="h-full overflow-y-auto p-4 md:p-6">
							{messages.length === 0 ? (
								<EmptyState
									isReady={isReady}
									onUpload={handleUpload}
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
							onUpload={handleUpload}
							isLoading={isQuerying}
							isUploading={isUploading || isIngesting}
							disabled={!canChat || isIngesting}
							disabledReason={
								isIngesting
									? "Document indexing in progress…"
									: !isReady
										? "Waiting for retrieval pipeline…"
										: ingestionStatus?.status !== "complete"
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
