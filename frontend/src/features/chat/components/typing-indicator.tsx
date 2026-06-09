import { IconRobot } from "@tabler/icons-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function TypingIndicator() {
	return (
		<div className="flex items-center gap-3">
			<Avatar size="sm">
				<AvatarFallback>
					<IconRobot />
				</AvatarFallback>
			</Avatar>
			<Card className="py-3">
				<CardContent>
					<div className="flex gap-1.5">
						{[0, 1, 2].map((i) => (
							<Skeleton
								key={i}
								className="size-2 rounded-full"
								style={{ animationDelay: `${i * 150}ms` }}
							/>
						))}
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
