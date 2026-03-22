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
	onUpload,
	isUploading,
}: {
	isReady: boolean;
	onUpload: (files: File[]) => Promise<void>;
	isUploading: boolean;
}) {
	return (
		<div className="flex h-full items-center justify-center">
			<Empty className="max-w-md w-full rounded-2xl border border-dashed bg-muted/20 py-12">
				<EmptyHeader>
					<EmptyMedia variant="icon">
						<Cloud className="h-6 w-6 text-muted-foreground" />
					</EmptyMedia>
					<EmptyTitle>
						{isReady ? "Library Empty" : "Pipeline Loading…"}
					</EmptyTitle>
					<EmptyDescription>
						{isReady
							? "Upload your PDF documents to the library to start chatting with your AI."
							: "The retrieval service is initialising. Please wait a moment."}
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
