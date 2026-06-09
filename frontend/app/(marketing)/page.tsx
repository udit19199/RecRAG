import type { Metadata } from "next";
import { IconBolt, IconDatabase, IconSearch } from "@tabler/icons-react";

export const metadata: Metadata = {
	title: "RecRAG — Enterprise Retrieval, Simplified",
	description:
		"High-performance pipeline to ingest, vectorize, and query unstructured data with millisecond latency.",
};
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const features = [
	{
		icon: IconDatabase,
		title: "Ingest",
		description:
			"Connect seamlessly to cloud storage, databases, and APIs. We handle chunking and parsing automatically.",
	},
	{
		icon: IconBolt,
		title: "Vectorize",
		description:
			"Leverage state-of-the-art embedding models to convert your unstructured text into high-dimensional vectors.",
	},
	{
		icon: IconSearch,
		title: "Query",
		description:
			"Execute ultra-fast semantic search with hybrid filtering and advanced reranking capabilities.",
	},
];

export default function LandingPage() {
	return (
		<div className="mx-auto flex w-full max-w-5xl flex-col gap-24 px-4 py-24">
			<section className="flex max-w-3xl flex-col gap-6">
				<h1 className="balance text-5xl leading-tight font-bold tracking-tight text-foreground sm:text-6xl">
					Enterprise Retrieval, <br />
					Simplified.
				</h1>
				<p className="max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground">
					RecRAG provides a high-performance pipeline to ingest, vectorize, and
					query your unstructured data with millisecond latency.
				</p>
				<div className="flex items-center gap-4 pt-4">
					<Button asChild size="lg">
						<Link href="/onboarding">Initialize Workspace</Link>
					</Button>
					<Button asChild variant="outline" size="lg">
						<Link href="/chat">Go to App</Link>
					</Button>
				</div>
			</section>

			<section className="flex flex-col gap-6 pt-8">
				<Separator />
				<div className="grid gap-6 sm:grid-cols-3">
					{features.map((feature) => (
						<Card key={feature.title} size="sm">
							<CardHeader>
								<div className="flex size-10 items-center justify-center rounded-md border bg-muted/50">
									<feature.icon />
								</div>
								<CardTitle className="text-lg">{feature.title}</CardTitle>
							</CardHeader>
							<CardContent>
								<CardDescription className="text-pretty">
									{feature.description}
								</CardDescription>
							</CardContent>
						</Card>
					))}
				</div>
			</section>
		</div>
	);
}
