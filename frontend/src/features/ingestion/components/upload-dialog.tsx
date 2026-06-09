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
import { Switch } from "@/components/ui/switch";
import type { UploadOptions } from "@/lib/api/types";

interface UploadDialogProps {
	onUpload: (
		files: File[],
		options?: Pick<UploadOptions, "replace">,
	) => Promise<void>;
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
	const [replaceExisting, setReplaceExisting] = React.useState(false);

	const handleOpenChange = (nextOpen: boolean) => {
		setOpen(nextOpen);
		if (!nextOpen) {
			setReplaceExisting(false);
		}
	};

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const files = Array.from(event.target.files || []);
		if (files.length > 0) {
			void onUpload(files, replaceExisting ? { replace: true } : undefined);
			setOpen(false);
			setReplaceExisting(false);
			if (fileInputRef.current) fileInputRef.current.value = "";
		}
	};

	return (
		<AlertDialog open={open} onOpenChange={handleOpenChange}>
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
						<FilePlus className="size-5 text-primary" />
						Upload Documents
					</AlertDialogTitle>
					<AlertDialogDescription>
						Add PDF documents to your library. You can upload more files at any
						time.
					</AlertDialogDescription>
				</AlertDialogHeader>

				<div className="space-y-4 py-4">
					<div className="rounded-lg border bg-muted/30 p-4">
						<h4 className="flex items-center gap-2 text-xs font-semibold text-foreground">
							<Info className="size-4" />
							Guidelines & Tips
						</h4>
						<ul className="mt-3 space-y-2 text-xs text-muted-foreground">
							<li className="flex items-start gap-2">
								<span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Only <strong>PDF files</strong> are supported for ingestion.
								</span>
							</li>
							<li className="flex items-start gap-2">
								<span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Select a <strong>Vision Model</strong> above to enable
									vision-assisted extraction for scanned documents and
									infographics.
								</span>
							</li>
							<li className="flex items-start gap-2">
								<span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
								<span>
									Keep individual files under <strong>50 MB</strong>.
								</span>
							</li>
						</ul>
					</div>

					<div className="flex items-center justify-between gap-3 rounded-lg border bg-background px-4 py-3">
						<div className="min-w-0">
							<p className="text-sm font-medium text-foreground">
								Replace existing documents
							</p>
							<p className="mt-0.5 text-xs text-muted-foreground">
								{replaceExisting
									? "The current library will be cleared before indexing."
									: "New files will be added to your existing library."}
							</p>
						</div>
						<Switch
							checked={replaceExisting}
							onCheckedChange={setReplaceExisting}
							disabled={isUploading}
							aria-label="Replace existing documents"
						/>
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
							aria-label="Select PDF files to upload"
							className="hidden"
							onChange={handleFileChange}
						/>
						<Button
							className="w-full"
							onClick={() => fileInputRef.current?.click()}
							disabled={isUploading}
						>
							{isUploading ? "Uploading…" : "Select Files & Start Ingestion"}
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
