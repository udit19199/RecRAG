import type { Metadata } from "next";
import "./globals.css";
import { JetBrains_Mono } from "next/font/google";
import { AppShell } from "@/components/layout/app-shell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

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
		<html lang="en" className={cn("dark", "font-mono", jetbrainsMono.variable)}>
			<body className="min-h-screen bg-background text-foreground antialiased">
				<TooltipProvider>
					<AppShell>{children}</AppShell>
				</TooltipProvider>
			</body>
		</html>
	);
}
