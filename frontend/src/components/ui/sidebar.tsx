"use client"

import * as React from "react"

const SIDEBAR_WIDTH = "18rem"

type SidebarContextType = {
  open: boolean
  setOpen: (v: boolean) => void
  toggleSidebar: () => void
}

const SidebarContext = React.createContext<SidebarContextType | null>(null)

export function SidebarProvider({ children, defaultOpen = true }: {
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = React.useState<boolean>(defaultOpen)

  const toggleSidebar = React.useCallback(() => setOpen((s) => !s), [])

  return (
    <SidebarContext.Provider value={{ open, setOpen, toggleSidebar }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const ctx = React.useContext(SidebarContext)
  if (!ctx) throw new Error('useSidebar must be used within SidebarProvider')
  return ctx
}

export function Sidebar({ children, className = "" }: React.ComponentProps<'div'>) {
  const { open } = React.useContext(SidebarContext) ?? { open: true }
  return (
    <aside
      data-sidebar
      className={`flex-shrink-0 transition-all duration-200 ease-in-out ${open ? 'w-[18rem]' : 'w-0'} overflow-hidden ${className}`}
      style={{ ['--sidebar-width' as any]: SIDEBAR_WIDTH }}
    >
      <div className="h-full min-h-0 w-[18rem] overflow-hidden">
        {children}
      </div>
    </aside>
  )
}

export function SidebarHeader({ children }: { children?: React.ReactNode }) {
  return (
    <div className="sticky top-0 z-10 border-b bg-background p-4">
      {children}
    </div>
  )
}

export function SidebarContent({ children }: { children?: React.ReactNode }) {
  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto p-4">
      {children}
    </div>
  )
}

export function SidebarFooter({ children }: { children?: React.ReactNode }) {
  return (
    <div className="sticky bottom-0 z-10 border-t bg-background p-4">
      {children}
    </div>
  )
}

export function SidebarGroup({ children }: { children?: React.ReactNode }) {
  return <div className="mb-4">{children}</div>
}

export function SidebarTrigger() {
  const { toggleSidebar } = useSidebar()
  return (
    <button onClick={toggleSidebar} aria-label="Toggle sidebar" className="rounded-md p-2 hover:bg-muted">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="1.5" y="1.5" width="13" height="13" rx="2" />
        <line x1="5.5" y1="1.5" x2="5.5" y2="14.5" />
      </svg>
    </button>
  )
}

export default Sidebar
