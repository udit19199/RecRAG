import type * as React from "react";

import { cn } from "@/lib/utils";

interface ProgressProps extends React.ComponentProps<"progress"> {
	value: number;
	max?: number;
}

function Progress({ className, value, max = 100, ...props }: ProgressProps) {
	const clampedValue = Math.min(max, Math.max(0, value));

	return (
		<progress
			value={clampedValue}
			max={max}
			className={cn(
				"h-2 w-full overflow-hidden rounded-full bg-primary/20 accent-primary [&::-moz-progress-bar]:rounded-full [&::-moz-progress-bar]:bg-primary [&::-webkit-progress-bar]:rounded-full [&::-webkit-progress-bar]:bg-primary/20 [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-value]:bg-primary",
				className,
			)}
			{...props}
		/>
	);
}

export { Progress };
