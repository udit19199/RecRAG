"use client";

import {
	IconCircleCheck,
	IconCircleX,
	IconClock,
	IconEye,
	IconLetterT,
} from "@tabler/icons-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import type { IngestionStatus } from "@/lib/api";

interface IngestionStatusDisplayProps {
	status: IngestionStatus | null;
	error: string | null;
	isLoading: boolean;
}

export default function IngestionStatusDisplay({
	status,
	error,
	isLoading,
}: IngestionStatusDisplayProps) {
	if (isLoading && !status) {
		return (
			<Card>
				<CardContent className="flex items-center gap-3 text-muted-foreground">
					<Spinner />
					<span className="text-sm font-medium">Loading status…</span>
				</CardContent>
			</Card>
		);
	}

	if (error) {
		return (
			<Alert variant="destructive">
				<IconCircleX />
				<AlertTitle>Status unavailable</AlertTitle>
				<AlertDescription>{error}</AlertDescription>
			</Alert>
		);
	}

	if (!status) return null;

	const statusConfig = {
		idle: {
			label: "Idle",
			description:
				"Upload your full PDF batch to build the corpus before chatting.",
			icon: <IconClock />,
		},
		processing: {
			label: "Processing",
			description: `Started at ${status.started_at || "unknown"}. Rebuilding the corpus and preparing chat...`,
			icon: <Spinner />,
		},
		complete: {
			label: "Complete",
			description: `Successfully processed ${status.files_processed || 0} files. Processing complete, you can now ask questions.`,
			icon: <IconCircleCheck />,
		},
		error: {
			label: "Error",
			description: status.error_message || "An unknown error occurred.",
			icon: <IconCircleX />,
		},
	} as const;

	const config = statusConfig[status.status];

	return (
		<Card
			className={status.status === "processing" ? "bg-muted/40" : undefined}
		>
			<CardContent className="flex items-start gap-3">
				<div className="mt-0.5 shrink-0 text-muted-foreground">
					{config.icon}
				</div>
				<div className="min-w-0">
					<Badge variant="secondary">{config.label}</Badge>
					<p className="mt-2 text-sm leading-relaxed text-foreground">
						{config.description}
					</p>
					{status.status === "processing" ? (
						<Skeleton
							className="mt-3 h-2 w-full rounded-full"
							aria-label="Indexing in progress"
						/>
					) : null}
					{status.completed_at ? (
						<p className="mt-1 text-xs text-muted-foreground">
							Completed at: {status.completed_at}
						</p>
					) : null}
					{status.extraction_mode && status.status !== "idle" ? (
						<div className="mt-2 flex items-center gap-1.5">
							{status.extraction_mode === "vision_assisted" ? (
								<IconEye className="size-3.5 text-muted-foreground" />
							) : (
								<IconLetterT className="size-3.5 text-muted-foreground" />
							)}
							<span className="text-xs text-muted-foreground">
								{status.extraction_mode === "vision_assisted"
									? "Vision Assisted"
									: "Text Only"}
							</span>
						</div>
					) : null}
				</div>
			</CardContent>
		</Card>
	);
}
