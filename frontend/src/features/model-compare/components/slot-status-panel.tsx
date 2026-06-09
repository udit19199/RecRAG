import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import type {
	SlotKey,
	SlotState,
} from "@/features/model-compare/components/model-comparison-types";

export function SlotStatusPanel({
	slotKey,
	state,
	isIngesting,
	ingestingSlot,
	onIngest,
}: {
	slotKey: SlotKey;
	state: SlotState;
	isIngesting: boolean;
	ingestingSlot: SlotKey | null;
	onIngest: () => void;
}) {
	if (state.isChecking || !state.embedding || !state.llm) {
		return null;
	}

	if (isIngesting && ingestingSlot === slotKey) {
		return (
			<Card size="sm" className="mt-4 bg-muted/50">
				<CardContent className="flex items-center gap-2 text-sm text-muted-foreground">
					<Spinner />
					Processing documents…
				</CardContent>
			</Card>
		);
	}

	if (!state.hasDocuments) {
		return (
			<Alert variant="destructive" className="mt-4">
				<AlertTitle>Index not found</AlertTitle>
				<AlertDescription className="flex flex-col gap-2">
					<span>
						This combination of Vision + Embedding models hasn't been used to
						index your files yet.
					</span>
					<Button
						type="button"
						onClick={onIngest}
						disabled={isIngesting}
						size="sm"
						className="self-start"
					>
						Process Documents for Slot {slotKey}
					</Button>
				</AlertDescription>
			</Alert>
		);
	}

	return null;
}
