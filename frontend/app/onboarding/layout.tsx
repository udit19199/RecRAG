import Link from "next/link";
import type { ReactNode } from "react";

export default function OnboardingLayout({
	children,
}: {
	children: ReactNode;
}) {
	return (
		<div className="min-h-screen bg-background flex flex-col">
			<header className="absolute top-0 left-0 w-full p-6 flex justify-between items-center z-10">
				<Link
					href="/"
					className="flex items-center gap-2 text-foreground font-bold text-xl tracking-tight"
				>
					<div className="w-6 h-6 bg-primary rounded-sm"></div>
					RecRAG
				</Link>
				<Link
					href="/chat"
					className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors px-4 py-2 rounded-md hover:bg-muted/50"
				>
					Skip Setup
				</Link>
			</header>
			<main className="flex-1 flex flex-col items-center justify-center p-4">
				{children}
			</main>
		</div>
	);
}
