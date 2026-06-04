import { Collapsible as CollapsiblePrimitive } from "radix-ui";
import type * as React from "react";

function CollapsibleTrigger({
	...props
}: React.ComponentProps<typeof CollapsiblePrimitive.CollapsibleTrigger>) {
	return (
		<CollapsiblePrimitive.CollapsibleTrigger
			data-slot="collapsible-trigger"
			{...props}
		/>
	);
}

export { CollapsibleTrigger };
