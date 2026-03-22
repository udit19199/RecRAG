import { Robot } from "@phosphor-icons/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge } from "@/components/ui/badge";
import { SourceContext } from "@/features/chat/components/source-context";
import type { ChatMessage } from "@/features/chat/types";

export function ChatMessageView({ message }: { message: ChatMessage }) {
	if (message.role === "user") {
		return (
			<div className="flex justify-end">
				<div className="max-w-[75%] rounded-xl border bg-muted px-4 py-3 text-sm text-foreground">
					{message.content}
				</div>
			</div>
		);
	}

	return (
		<div className="flex items-start gap-3">
			<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground">
				<Robot className="h-4 w-4" />
			</div>
			<div className="min-w-0 flex-1 space-y-3">
				{message.error ? (
					<div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
						{message.error}
					</div>
				) : (
					<div className="rounded-xl border bg-background px-4 py-3">
						<div className="prose prose-sm max-w-none text-foreground [&_a]:text-primary [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-sm [&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_li]:text-foreground [&_ol]:text-foreground [&_p]:text-foreground [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-4 [&_strong]:text-foreground [&_ul]:text-foreground">
							<ReactMarkdown remarkPlugins={[remarkGfm]}>
								{message.content}
							</ReactMarkdown>
						</div>
					</div>
				)}

				{message.context?.length ? (
					<SourceContext context={message.context} />
				) : null}

				{message.eval ? (
					<div className="rounded-xl border bg-background px-4 py-3">
						<p className="mb-2 text-xs font-medium text-muted-foreground">
							Evaluation
						</p>
						<div className="flex flex-wrap gap-2">
							{Object.entries(message.eval).map(([key, value]) => (
								<Badge key={key} variant="secondary" className="gap-2">
									<span>{key}</span>
									<span className="text-muted-foreground">
										{value.toFixed(2)}
									</span>
								</Badge>
							))}
						</div>
					</div>
				) : null}
			</div>
		</div>
	);
}
