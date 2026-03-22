import { type FormEvent, type KeyboardEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { UploadDialog } from "@/features/ingestion/components/upload-dialog";
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
	onUpload: (files: File[]) => Promise<void>;
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
			<div
				className={cn(
					"flex items-end gap-3 rounded-xl border bg-background px-4 py-3 transition-all",
					!disabled &&
						"focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/20",
					disabled && "cursor-not-allowed bg-muted/20 opacity-60",
				)}
			>
				<UploadDialog
					onUpload={onUpload}
					isUploading={isUploading}
					trigger={
						<Button
							type="button"
							variant="ghost"
							size="icon"
							className="h-9 w-9 shrink-0 text-muted-foreground hover:text-foreground"
							disabled={isUploading}
							title="Upload PDFs"
						>
							<svg
								aria-hidden="true"
								className="h-5 w-5"
								fill="none"
								viewBox="0 0 24 24"
								strokeWidth={2}
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									d="M12 4.5v15m7.5-7.5h-15"
								/>
							</svg>
						</Button>
					}
				/>
				<textarea
					value={value}
					onChange={(event) => setValue(event.target.value)}
					onKeyDown={handleKeyDown}
					placeholder={
						disabled
							? disabledReason || "Chat disabled"
							: "Ask a question about your documents..."
					}
					rows={1}
					disabled={disabled}
					className="max-h-36 min-h-[1.5rem] flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed"
					onInput={(event) => {
						const el = event.currentTarget;
						el.style.height = "auto";
						el.style.height = `${el.scrollHeight}px`;
					}}
				/>
				<Button
					type="submit"
					disabled={!value.trim() || isLoading || disabled}
					size="icon"
				>
					{isLoading ? (
						<svg
							aria-hidden="true"
							className="h-4 w-4 animate-spin"
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
					) : (
						<svg
							aria-hidden="true"
							className="h-4 w-4"
							fill="none"
							viewBox="0 0 24 24"
							strokeWidth={2}
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
							/>
						</svg>
					)}
				</Button>
			</div>
			<p className="mt-1.5 text-center text-[10px] text-muted-foreground">
				Enter to send, Shift+Enter for newline
			</p>
		</form>
	);
}
