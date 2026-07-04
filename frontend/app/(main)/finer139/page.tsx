import type { Metadata } from "next";
import { Finer139Workbench } from "@/features/finer139/components/finer139-workbench";

export const metadata: Metadata = {
	title: "FiNER-139 — RecRAG",
	description:
		"Compare graph-construction methods on financial entity recognition using the FiNER-139 dataset.",
};

export default function Finer139Page() {
	return <Finer139Workbench />;
}
