"use client";

import { Eye, TextT } from "@phosphor-icons/react";
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
			<div className="rounded-lg border border-border bg-card px-4 py-3">
				<div className="flex items-center gap-3 text-muted-foreground">
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
					<span className="text-sm font-medium">Loading status…</span>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3">
				<div className="flex items-start gap-3">
					<svg
						aria-hidden="true"
						className="mt-0.5 size-5 shrink-0 text-destructive"
						viewBox="0 0 20 20"
						fill="currentColor"
					>
						<path
							fillRule="evenodd"
							d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
							clipRule="evenodd"
						/>
					</svg>
					<div>
						<p className="text-sm font-medium text-destructive">
							Status unavailable
						</p>
						<p className="mt-1 text-xs leading-relaxed text-muted-foreground">
							{error}
						</p>
					</div>
				</div>
			</div>
		);
	}

	if (!status) return null;

	const statusConfig = {
		idle: {
			label: "Idle",
			description:
				"Upload your full PDF batch to build the corpus before chatting.",
			color: "text-muted-foreground",
			icon: (
				<svg
					aria-hidden="true"
					className="size-5"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						fillRule="evenodd"
						d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
						clipRule="evenodd"
					/>
				</svg>
			),
		},
		processing: {
			label: "Processing",
			description: `Started at ${status.started_at || "unknown"}. Rebuilding the corpus and preparing chat...`,
			color: "text-primary",
			icon: (
				<svg
					aria-hidden="true"
					className="size-5 animate-spin"
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
			),
		},
		complete: {
			label: "Complete",
			description: `Successfully processed ${status.files_processed || 0} files. Processing complete, you can now ask questions.`,
			color: "text-success dark:text-success-foreground",
			icon: (
				<svg
					aria-hidden="true"
					className="size-5"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						fillRule="evenodd"
						d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
						clipRule="evenodd"
					/>
				</svg>
			),
		},
		error: {
			label: "Error",
			description: status.error_message || "An unknown error occurred.",
			color: "text-destructive",
			icon: (
				<svg
					aria-hidden="true"
					className="size-5"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						fillRule="evenodd"
						d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
						clipRule="evenodd"
					/>
				</svg>
			),
		},
	} as const;

	const config = statusConfig[status.status];
	const containerClass =
		status.status === "processing"
			? "border-border bg-muted/40"
			: "border-border bg-card";

	return (
		<div className={`rounded-lg border px-4 py-3 ${containerClass}`}>
			<div className="flex items-start gap-3">
				<div className={`mt-0.5 flex-shrink-0 ${config.color}`}>
					{config.icon}
				</div>
				<div className="min-w-0">
					<span
						className={`inline-flex rounded-full border border-border bg-muted px-2 py-0.5 text-xs font-medium ${config.color}`}
					>
						{config.label}
					</span>
					<p className="mt-2 text-sm leading-relaxed text-foreground">
						{config.description}
					</p>
					{status.completed_at ? (
						<p className="mt-1 text-xs text-muted-foreground">
							Completed at: {status.completed_at}
						</p>
					) : null}
					{status.extraction_mode && status.status !== "idle" ? (
						<div className="mt-2 flex items-center gap-1.5">
							{status.extraction_mode === "vision_assisted" ? (
								<Eye className="size-3.5 text-muted-foreground" />
							) : (
								<TextT className="size-3.5 text-muted-foreground" />
							)}
							<span className="text-xs text-muted-foreground">
								{status.extraction_mode === "vision_assisted"
									? "Vision Assisted"
									: "Text Only"}
							</span>
						</div>
					) : null}
				</div>
			</div>
		</div>
	);
}
