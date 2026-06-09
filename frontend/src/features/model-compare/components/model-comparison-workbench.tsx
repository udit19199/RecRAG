"use client";

import { Alert, AlertDescription } from "@/components/ui/alert";
import {
	Card,
	CardContent,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Empty,
	EmptyDescription,
	EmptyHeader,
	EmptyMedia,
	EmptyTitle,
} from "@/components/ui/empty";
import { ChatInput } from "@/features/chat/components/chat-input";
import { ComparisonSlotColumn } from "@/features/model-compare/components/comparison-slot-column";
import { useModelComparison } from "@/features/model-compare/hooks/use-model-comparison";
import { SlotPicker } from "./slot-picker";

export function ModelComparisonWorkbench() {
	const {
		state,
		isLoading,
		canChat,
		chatBottomRef,
		handleSlotVisionChange,
		handleSlotEmbedChange,
		handleSlotLlmChange,
		handleIngest,
		handleQuery,
	} = useModelComparison();

	const slotsReady =
		state.slotA.llm &&
		state.slotB.llm &&
		state.slotA.embedding &&
		state.slotB.embedding;

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30 p-4">
			<div className="mx-auto flex h-full w-full max-w-[1440px] flex-col gap-4">
				<Card className="shrink-0">
					<CardHeader>
						<div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
							<div>
								<CardTitle className="text-xl">Model Comparison</CardTitle>
								<p className="mt-1 text-sm text-muted-foreground">
									Select two configurations to compare their generation quality
									side-by-side.
								</p>
							</div>
							{state.error ? (
								<Alert variant="destructive">
									<AlertDescription>{state.error}</AlertDescription>
								</Alert>
							) : null}
						</div>
					</CardHeader>
					<CardContent className="grid gap-6 md:grid-cols-2">
						<SlotPicker
							label="Slot A Configuration"
							providers={state.providers}
							isLoading={isLoading}
							disabled={state.isQuerying || state.isIngesting}
							visionValue={state.slotA.vision}
							embedValue={state.slotA.embedding}
							llmValue={state.slotA.llm}
							onVisionChange={(val) => handleSlotVisionChange("A", val)}
							onEmbedChange={(val) => handleSlotEmbedChange("A", val)}
							onLlmChange={(val) => handleSlotLlmChange("A", val)}
						/>
						<SlotPicker
							label="Slot B Configuration"
							providers={state.providers}
							isLoading={isLoading}
							disabled={state.isQuerying || state.isIngesting}
							visionValue={state.slotB.vision}
							embedValue={state.slotB.embedding}
							llmValue={state.slotB.llm}
							onVisionChange={(val) => handleSlotVisionChange("B", val)}
							onEmbedChange={(val) => handleSlotEmbedChange("B", val)}
							onLlmChange={(val) => handleSlotLlmChange("B", val)}
						/>
					</CardContent>
				</Card>

				{!slotsReady ? (
					<Card className="flex flex-1 items-center justify-center p-8">
						<Empty>
							<EmptyHeader>
								<EmptyMedia variant="icon">{"//"}</EmptyMedia>
								<EmptyTitle>Select complete configurations</EmptyTitle>
								<EmptyDescription>
									Use the dropdowns above to choose the Vision, Embedding, and
									LLM for both slots.
								</EmptyDescription>
							</EmptyHeader>
						</Empty>
					</Card>
				) : (
					<div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden">
						<div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden md:grid-cols-2">
							<ComparisonSlotColumn
								slotKey="A"
								label="Slot A"
								state={state.slotA}
								messages={state.messagesA}
								isQuerying={state.isQuerying}
								isIngesting={state.isIngesting}
								ingestingSlot={state.ingestingSlot}
								onIngest={() => handleIngest("A")}
								scrollAnchorRef={chatBottomRef}
							/>
							<ComparisonSlotColumn
								slotKey="B"
								label="Slot B"
								state={state.slotB}
								messages={state.messagesB}
								isQuerying={state.isQuerying}
								isIngesting={state.isIngesting}
								ingestingSlot={state.ingestingSlot}
								onIngest={() => handleIngest("B")}
							/>
						</div>

						<Card className="shrink-0">
							<CardFooter className="border-0 bg-transparent p-4">
								<ChatInput
									onSubmit={handleQuery}
									onUpload={async () => {}}
									isLoading={state.isQuerying}
									isUploading={false}
									disabled={!canChat || state.isQuerying}
									disabledReason={
										state.isIngesting
											? "Document processing in progress…"
											: !canChat
												? "Make sure both slots have documents processed."
												: undefined
									}
								/>
							</CardFooter>
						</Card>
					</div>
				)}
			</div>
		</div>
	);
}
