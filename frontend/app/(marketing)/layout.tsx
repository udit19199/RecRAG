import Link from "next/link";
import type { ReactNode } from "react";

export default function MarketingLayout({ children }: { children: ReactNode }) {
	return (
		<div className="min-h-screen bg-background flex flex-col font-sans">
			<header className="w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
				<div className="container mx-auto px-4 h-14 flex items-center justify-between">
					<Link
						href="/"
						className="flex items-center gap-2 text-foreground font-bold text-sm tracking-tight"
					>
						<div className="w-4 h-4 bg-foreground rounded-sm"></div>
						RecRAG
					</Link>

					<nav className="flex items-center gap-6 text-sm font-medium">
						<Link
							href="/docs"
							className="text-muted-foreground hover:text-foreground transition-colors"
						>
							Documentation
						</Link>
						<Link
							href="/github"
							className="text-muted-foreground hover:text-foreground transition-colors"
						>
							GitHub
						</Link>
					</nav>
				</div>
			</header>
			<main className="flex-1 flex flex-col items-center">{children}</main>
		</div>
	);
}
