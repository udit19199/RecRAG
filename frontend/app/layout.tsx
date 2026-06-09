import type { Metadata } from "next";
import "./globals.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const inter = Inter({
	subsets: ["latin"],
	variable: "--font-sans",
});

const jetbrainsMono = JetBrains_Mono({
	subsets: ["latin"],
	variable: "--font-mono",
});

export const metadata: Metadata = {
	title: "RecRAG - RAG Pipeline",
	description: "Retrieval-Augmented Generation system for querying documents",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html
			lang="en"
			className={cn(
				"dark",
				"font-sans",
				inter.variable,
				jetbrainsMono.variable,
			)}
		>
			<body className="min-h-screen bg-background text-foreground antialiased">
				<TooltipProvider>{children}</TooltipProvider>
			</body>
		</html>
	);
}
