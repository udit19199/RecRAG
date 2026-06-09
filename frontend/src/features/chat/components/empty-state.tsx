import { Cloud } from "@phosphor-icons/react";
import {
	Empty,
	EmptyContent,
	EmptyDescription,
	EmptyHeader,
	EmptyMedia,
	EmptyTitle,
} from "@/components/ui/empty";
import { UploadDialog } from "@/features/ingestion/components/upload-dialog";

export function EmptyState({
	isReady,
	error,
	onUpload,
	isUploading,
}: {
	isReady: boolean;
	error?: string | null;
	onUpload: (files: File[]) => Promise<void>;
	isUploading: boolean;
}) {
	return (
		<div className="flex h-full items-center justify-center">
			<Empty className="max-w-md w-full rounded-2xl border border-dashed bg-muted/20 py-12">
				<EmptyHeader>
					<EmptyMedia variant="icon">
						<Cloud className="size-6 text-muted-foreground" />
					</EmptyMedia>
					<EmptyTitle>
						{isReady
							? "Library Empty"
							: error
								? "Configuration Error"
								: "Pipeline Loading…"}
					</EmptyTitle>
					<EmptyDescription>
						{isReady ? (
							"Upload your PDF documents to the library to start chatting with your AI."
						) : error ? (
							<span className="text-destructive font-medium">{error}</span>
						) : (
							"The retrieval service is initialising. Please wait a moment."
						)}
					</EmptyDescription>
				</EmptyHeader>
				{isReady ? (
					<EmptyContent>
						<UploadDialog onUpload={onUpload} isUploading={isUploading} />
					</EmptyContent>
				) : null}
			</Empty>
		</div>
	);
}
