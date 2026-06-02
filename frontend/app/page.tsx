import type { Metadata } from "next";
import { ChatWorkbench } from "@/features/chat/components/chat-workbench";

export const metadata: Metadata = {
	title: "RecRAG — Retrieval-Augmented Generation",
	description: "Upload documents and chat with an AI that retrieves relevant context from your files.",
};

export default function HomePage() {
	return <ChatWorkbench />;
}
