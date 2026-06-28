"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import FileUploader from "@/features/ingestion/components/file-uploader";
import IngestionStatusDisplay from "@/features/ingestion/components/ingestion-status";
import { useCorpusUpload } from "@/features/ingestion/hooks/use-corpus-upload";
import { createRun } from "@/lib/api/orchestrator";

type SubmitResult = { ok: true; runId: string } | { ok: false; error: string };

async function submitRecommendation(query: string): Promise<SubmitResult> {
	try {
		const { run_id } = await createRun({ use_case: query });
		return { ok: true, runId: run_id };
	} catch (err) {
		return {
			ok: false,
			error:
				err instanceof Error ? err.message : "Failed to start recommendation",
		};
	}
}

export function GenerateRecommendationForm() {
	const router = useRouter();
	const [query, setQuery] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const {
		uploadedFiles,
		ingestionStatus,
		isUploading,
		uploadProgress,
		feedback,
		upload,
	} = useCorpusUpload();

	const onSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		const trimmed = query.trim();
		if (!trimmed) {
			setError("Enter a query describing your use case.");
			return;
		}

		setLoading(true);
		setError(null);
		const result = await submitRecommendation(trimmed);
		setLoading(false);

		if (!result.ok) {
			setError(result.error);
			return;
		}

		router.push(`/recommendations/${result.runId}`);
	};

	const isIndexing = ingestionStatus?.status === "processing";

	return (
		<div className="flex flex-col gap-8">
			<section className="flex flex-col gap-4">
				<h2 className="text-sm font-medium">Corpus (optional)</h2>
				<p className="text-sm text-muted-foreground">
					Skip upload to get a preliminary recommendation from your query only.
					Upload PDFs to benchmark pipelines on your documents and unlock
					export.
				</p>
				<FileUploader
					onUpload={upload}
					isUploading={isUploading}
					disabled={loading || isIndexing}
				/>
				{isUploading && uploadProgress !== null ? (
					<Progress value={uploadProgress} className="h-2" />
				) : null}
				{feedback ? (
					<Alert
						variant={feedback.type === "error" ? "destructive" : "default"}
					>
						<AlertDescription>{feedback.message}</AlertDescription>
					</Alert>
				) : null}
				<IngestionStatusDisplay
					status={ingestionStatus}
					error={null}
					isLoading={false}
				/>
				{uploadedFiles.length > 0 ? (
					<ul className="text-sm text-muted-foreground">
						{uploadedFiles.map((file) => (
							<li key={file}>{file}</li>
						))}
					</ul>
				) : null}
			</section>

			<form onSubmit={onSubmit} className="flex flex-col gap-4">
				<h2 className="text-sm font-medium">Query</h2>
				<Textarea
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="e.g. A law firm needs employees to query policy PDFs with citations. Budget around $1,500/month."
					rows={6}
					required
					disabled={loading}
				/>
				{error ? <p className="text-sm text-destructive">{error}</p> : null}
				<Button type="submit" disabled={loading || isUploading || isIndexing}>
					{loading ? "Starting…" : "Generate"}
				</Button>
			</form>
		</div>
	);
}
