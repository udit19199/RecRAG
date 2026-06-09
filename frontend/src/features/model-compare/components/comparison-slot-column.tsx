import type { RefObject } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatMessageView } from "@/features/chat/components/chat-message";
import { TypingIndicator } from "@/features/chat/components/typing-indicator";
import type { ChatMessage } from "@/features/chat/types";
import type {
	SlotKey,
	SlotState,
} from "@/features/model-compare/components/model-comparison-types";
import { SlotStatusPanel } from "@/features/model-compare/components/slot-status-panel";

export function ComparisonSlotColumn({
	slotKey,
	label,
	state,
	messages,
	isQuerying,
	isIngesting,
	ingestingSlot,
	onIngest,
	scrollAnchorRef,
}: {
	slotKey: SlotKey;
	label: string;
	state: SlotState;
	messages: ChatMessage[];
	isQuerying: boolean;
	isIngesting: boolean;
	ingestingSlot: SlotKey | null;
	onIngest: () => void;
	scrollAnchorRef?: RefObject<HTMLDivElement | null>;
}) {
	return (
		<Card className="flex h-full flex-col overflow-y-auto py-0">
			<CardHeader className="sticky top-0 z-10 border-b bg-card pb-4">
				<CardTitle className="text-sm tracking-wider">{label}</CardTitle>
			</CardHeader>
			<CardContent className="flex flex-col gap-5">
				<SlotStatusPanel
					slotKey={slotKey}
					state={state}
					isIngesting={isIngesting}
					ingestingSlot={ingestingSlot}
					onIngest={onIngest}
				/>
				{messages.length === 0 ? (
					<p className="text-sm text-muted-foreground">No messages yet.</p>
				) : (
					messages.map((message) => (
						<ChatMessageView key={message.id} message={message} />
					))
				)}
				{isQuerying ? <TypingIndicator /> : null}
				{scrollAnchorRef ? <div ref={scrollAnchorRef} /> : null}
			</CardContent>
		</Card>
	);
}
