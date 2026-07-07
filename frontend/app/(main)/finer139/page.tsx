import { redirect } from "next/navigation";

/** Legacy route — merged into /graphrag?tab=benchmark */
export default function Finer139RedirectPage() {
	redirect("/graphrag?tab=benchmark");
}
