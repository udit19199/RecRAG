import { Robot } from "@phosphor-icons/react";

export function TypingIndicator() {
	return (
		<div className="flex items-center gap-3">
			<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground">
				<Robot className="h-4 w-4" />
			</div>
			<div className="rounded-xl border bg-muted/30 px-4 py-3">
				<div className="flex gap-1.5">
					{[0, 1, 2].map((i) => (
						<span
							key={i}
							className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"
							style={{ animationDelay: `${i * 150}ms` }}
						/>
					))}
				</div>
			</div>
		</div>
	);
}
