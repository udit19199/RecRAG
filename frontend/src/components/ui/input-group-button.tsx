import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const inputGroupButtonVariants = cva(
	"flex items-center gap-2 text-xs shadow-none",
	{
		variants: {
			size: {
				xs: "h-6 gap-1 rounded-none px-1.5 [&>svg:not([class*='size-'])]:size-3.5",
				sm: "gap-1",
				"icon-xs": "size-6 rounded-none p-0 has-[>svg]:p-0",
				"icon-sm": "size-7 p-0 has-[>svg]:p-0",
			},
		},
		defaultVariants: {
			size: "xs",
		},
	},
);

function InputGroupButton({
	className,
	type = "button",
	variant = "ghost",
	size = "xs",
	...props
}: Omit<React.ComponentProps<typeof Button>, "size"> &
	VariantProps<typeof inputGroupButtonVariants>) {
	return (
		<Button
			type={type}
			data-size={size}
			variant={variant}
			className={cn(inputGroupButtonVariants({ size }), className)}
			{...props}
		/>
	);
}

export { InputGroupButton };
