import { IconPlus, IconSend } from "@tabler/icons-react";
import { type FormEvent, type KeyboardEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import {
	InputGroup,
	InputGroupAddon,
	InputGroupTextarea,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { UploadDialog } from "@/features/ingestion/components/upload-dialog";
import type { UploadOptions } from "@/lib/api/types";
import { cn } from "@/lib/utils";

export function ChatInput({
	onSubmit,
	onUpload,
	isLoading,
	isUploading,
	disabled,
	disabledReason,
}: {
	onSubmit: (query: string) => Promise<void>;
	onUpload: (
		files: File[],
		options?: Pick<UploadOptions, "replace">,
	) => Promise<void>;
	isLoading: boolean;
	isUploading: boolean;
	disabled?: boolean;
	disabledReason?: string;
}) {
	const [value, setValue] = useState("");

	const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
		event.preventDefault();
		if (!value.trim() || isLoading || disabled) return;

		const query = value;
		setValue("");
		await onSubmit(query);
	};

	const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
		if (event.key === "Enter" && !event.shiftKey) {
			event.preventDefault();
			void handleSubmit(event as unknown as FormEvent<HTMLFormElement>);
		}
	};

	return (
		<form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
			<InputGroup
				className={cn(
					"h-auto min-h-12 items-end rounded-xl px-1 py-1",
					disabled && "cursor-not-allowed opacity-60",
				)}
			>
				<InputGroupAddon align="inline-start" className="pb-1.5">
					<UploadDialog
						onUpload={onUpload}
						isUploading={isUploading}
						trigger={
							<Button
								type="button"
								variant="ghost"
								size="icon-sm"
								disabled={isUploading}
								aria-label="Upload PDFs"
							>
								<IconPlus data-icon="inline-start" />
							</Button>
						}
					/>
				</InputGroupAddon>
				<InputGroupTextarea
					value={value}
					onChange={(event) => setValue(event.target.value)}
					onKeyDown={handleKeyDown}
					aria-label="Ask a question about your documents"
					placeholder={
						disabled
							? disabledReason || "Chat disabled"
							: "Ask a question about your documents..."
					}
					rows={1}
					disabled={disabled}
					className="max-h-36 min-h-[1.5rem] py-2"
					onInput={(event) => {
						const el = event.currentTarget;
						el.style.height = "auto";
						el.style.height = `${el.scrollHeight}px`;
					}}
				/>
				<InputGroupAddon align="inline-end" className="pb-1.5">
					<Button
						type="submit"
						disabled={!value.trim() || isLoading || disabled}
						size="icon-sm"
						aria-label={isLoading ? "Sending..." : "Send message"}
					>
						{isLoading ? <Spinner /> : <IconSend data-icon="inline-start" />}
					</Button>
				</InputGroupAddon>
			</InputGroup>
			<p className="mt-1.5 text-center text-[10px] text-muted-foreground">
				Enter to send, Shift+Enter for newline
			</p>
		</form>
	);
}
