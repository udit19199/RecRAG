import { IconRobot } from "@tabler/icons-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SourceContext } from "@/features/chat/components/source-context";
import type { ChatMessage } from "@/features/chat/types";

export function ChatMessageView({ message }: { message: ChatMessage }) {
	if (message.role === "user") {
		return (
			<div className="flex justify-end">
				<Card className="max-w-[75%] py-3">
					<CardContent className="text-sm">{message.content}</CardContent>
				</Card>
			</div>
		);
	}

	return (
		<div className="flex items-start gap-3">
			<Avatar size="sm">
				<AvatarFallback>
					<IconRobot />
				</AvatarFallback>
			</Avatar>
			<div className="flex min-w-0 flex-1 flex-col gap-3">
				{message.error ? (
					<Alert variant="destructive">
						<AlertDescription>{message.error}</AlertDescription>
					</Alert>
				) : (
					<Card className="py-3">
						<CardContent>
							<div className="prose prose-sm max-w-none text-foreground [&_a]:text-primary [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-sm [&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_li]:text-foreground [&_ol]:text-foreground [&_p]:text-foreground [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-4 [&_strong]:text-foreground [&_ul]:text-foreground">
								<ReactMarkdown remarkPlugins={[remarkGfm]}>
									{message.content}
								</ReactMarkdown>
							</div>
						</CardContent>
					</Card>
				)}

				{message.context?.length ? (
					<SourceContext context={message.context} />
				) : null}

				{message.eval ? (
					<Card size="sm">
						<CardHeader className="border-b pb-3">
							<CardTitle className="text-xs font-medium text-muted-foreground">
								Evaluation
							</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="flex flex-wrap gap-2">
								{Object.entries(message.eval).map(([key, value]) => (
									<Badge key={key} variant="secondary" className="gap-2">
										<span>{key}</span>
										<span className="text-muted-foreground">
											{value.toFixed(2)}
										</span>
									</Badge>
								))}
							</div>
						</CardContent>
					</Card>
				) : null}
			</div>
		</div>
	);
}
