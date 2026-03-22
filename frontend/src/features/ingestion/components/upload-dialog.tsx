"use client";

import { FilePlus, Info } from "@phosphor-icons/react";
import * as React from "react";
import {
	AlertDialog,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
	AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";

interface UploadDialogProps {
	onUpload: (files: File[]) => Promise<void>;
	isUploading: boolean;
	trigger?: React.ReactNode;
}

export function UploadDialog({
	onUpload,
	isUploading,
	trigger,
}: UploadDialogProps) {
	const fileInputRef = React.useRef<HTMLInputElement>(null);
	const [open, setOpen] = React.useState(false);

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = Array.from(event.target.files || []);
		if (files.length > 0) {
			void onUpload(files);
			setOpen(false);
			if (fileInputRef.current) fileInputRef.current.value = "";
		}
	};

	return (
		<AlertDialog open={open} onOpenChange={setOpen}>
			<AlertDialogTrigger asChild>
				{trigger || (
					<Button variant="outline" size="sm">
						Upload Files
					</Button>
				)}
			</AlertDialogTrigger>
			<AlertDialogContent className="max-w-md">
				<AlertDialogHeader>
					<AlertDialogTitle className="flex items-center gap-2">
						<FilePlus className="h-5 w-5 text-primary" />
						Upload Documents
					</AlertDialogTitle>
					<AlertDialogDescription>
						Choose your full PDF document set to build the retrieval corpus.
					</AlertDialogDescription>
				</AlertDialogHeader>

				<div className="space-y-4 py-4">
					<div className="rounded-lg border bg-muted/30 p-4">
						<h4 className="flex items-center gap-2 text-xs font-semibold text-foreground">
							<Info className="h-4 w-4" />
							Guidelines & Tips
						</h4>
						<ul className="mt-3 space-y-2 text-xs text-muted-foreground">
							<li className="flex items-start gap-2">
								<span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Only <strong>PDF files</strong> are supported for ingestion.
								</span>
							</li>
							<li className="flex items-start gap-2">
								<span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Use text-based PDFs (scanned images may not be readable).
								</span>
							</li>
							<li className="flex items-start gap-2">
								<span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Keep individual files under <strong>50 MB</strong>.
								</span>
							</li>
							<li className="flex items-start gap-2">
								<span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									A new upload will <strong>replace</strong> the entire current
									corpus.
								</span>
							</li>
						</ul>
					</div>

					<div className="flex flex-col gap-2">
						<p className="px-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
							Ready to process?
						</p>
						<input
							ref={fileInputRef}
							type="file"
							multiple
							accept=".pdf"
							className="hidden"
							onChange={handleFileChange}
						/>
						<Button
							className="w-full"
							onClick={() => fileInputRef.current?.click()}
							disabled={isUploading}
						>
							{isUploading ? "Uploading..." : "Select Files & Start Ingestion"}
						</Button>
					</div>
				</div>

				<AlertDialogFooter>
					<AlertDialogCancel disabled={isUploading}>Cancel</AlertDialogCancel>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	);
}
