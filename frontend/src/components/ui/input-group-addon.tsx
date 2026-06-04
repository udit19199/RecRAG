"use client";

import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";

import { cn } from "@/lib/utils";

const inputGroupAddonVariants = cva(
	"flex h-auto cursor-text items-center justify-center gap-2 py-1.5 text-xs font-medium text-muted-foreground select-none group-data-[disabled=true]/input-group:opacity-50 [&>kbd]:rounded-none [&>svg:not([class*='size-'])]:size-4",
	{
		variants: {
			align: {
				"inline-start":
					"order-first pl-2 has-[>button]:ml-[-0.3rem] has-[>kbd]:ml-[-0.15rem]",
				"inline-end":
					"order-last pr-2 has-[>button]:mr-[-0.3rem] has-[>kbd]:mr-[-0.15rem]",
				"block-start":
					"order-first w-full justify-start px-2.5 pt-2 group-has-[>input]/input-group:pt-2 [.border-b]:pb-2",
				"block-end":
					"order-last w-full justify-start px-2.5 pb-2 group-has-[>input]/input-group:pb-2 [.border-t]:pt-2",
			},
		},
		defaultVariants: {
			align: "inline-start",
		},
	},
);

function InputGroupAddon({
	className,
	align = "inline-start",
	...props
}: React.ComponentProps<"div"> & VariantProps<typeof inputGroupAddonVariants>) {
	return (
		// biome-ignore lint/a11y/noStaticElementInteractions: click-to-focus layout wrapper for input group
		<div
			data-slot="input-group-addon"
			data-align={align}
			className={cn(inputGroupAddonVariants({ align }), className)}
			onClick={(event) => {
				if ((event.target as HTMLElement).closest("button")) {
					return;
				}
				event.currentTarget.parentElement?.querySelector("input")?.focus();
			}}
			onKeyDown={(event) => {
				if (event.key === "Enter" || event.key === " ") {
					event.preventDefault();
					if ((event.target as HTMLElement).closest("button")) {
						return;
					}
					event.currentTarget.parentElement?.querySelector("input")?.focus();
				}
			}}
			{...props}
		/>
	);
}

export { InputGroupAddon };
