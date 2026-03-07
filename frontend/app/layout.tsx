import type { Metadata } from "next";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

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
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <TooltipProvider>
          <main className="min-h-screen">
            {children}
          </main>
        </TooltipProvider>
      </body>
    </html>
  );
}
