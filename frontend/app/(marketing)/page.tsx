import { Database, Search, Zap } from "lucide-react";
import Link from "next/link";

export default function LandingPage() {
	return (
		<div className="w-full flex flex-col max-w-5xl mx-auto px-4 py-24 gap-24">
			{/* Hero */}
			<section className="flex flex-col gap-6 max-w-3xl">
				<h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-foreground balance leading-tight">
					Enterprise Retrieval, <br />
					Simplified.
				</h1>
				<p className="text-lg text-muted-foreground max-w-xl text-pretty leading-relaxed">
					RecRAG provides a high-performance pipeline to ingest, vectorize, and
					query your unstructured data with millisecond latency.
				</p>
				<div className="flex items-center gap-4 pt-4">
					<Link
						href="/onboarding"
						className="inline-flex h-10 items-center justify-center rounded-md bg-foreground px-8 text-sm font-medium text-background transition-colors hover:bg-foreground/90"
					>
						Initialize Workspace
					</Link>
					<Link
						href="/chat"
						className="inline-flex h-10 items-center justify-center rounded-md border border-border bg-background px-8 text-sm font-medium text-foreground transition-colors hover:bg-muted/50"
					>
						Go to App
					</Link>
				</div>
			</section>

			{/* Features Grid */}
			<section className="grid sm:grid-cols-3 gap-6 pt-8 border-t border-border">
				<div className="flex flex-col gap-3">
					<div className="w-10 h-10 border border-border flex items-center justify-center rounded-md bg-muted/50">
						<Database className="w-5 h-5 text-foreground" />
					</div>
					<h3 className="font-semibold text-foreground text-lg">Ingest</h3>
					<p className="text-sm text-muted-foreground text-pretty">
						Connect seamlessly to cloud storage, databases, and APIs. We handle
						chunking and parsing automatically.
					</p>
				</div>
				<div className="flex flex-col gap-3">
					<div className="w-10 h-10 border border-border flex items-center justify-center rounded-md bg-muted/50">
						<Zap className="w-5 h-5 text-foreground" />
					</div>
					<h3 className="font-semibold text-foreground text-lg">Vectorize</h3>
					<p className="text-sm text-muted-foreground text-pretty">
						Leverage state-of-the-art embedding models to convert your
						unstructured text into high-dimensional vectors.
					</p>
				</div>
				<div className="flex flex-col gap-3">
					<div className="w-10 h-10 border border-border flex items-center justify-center rounded-md bg-muted/50">
						<Search className="w-5 h-5 text-foreground" />
					</div>
					<h3 className="font-semibold text-foreground text-lg">Query</h3>
					<p className="text-sm text-muted-foreground text-pretty">
						Execute ultra-fast semantic search with hybrid filtering and
						advanced reranking capabilities.
					</p>
				</div>
			</section>
		</div>
	);
}
