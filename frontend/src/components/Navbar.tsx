'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

/**
 * Navigation link configuration
 */
const navLinks = [
  { href: '/', label: 'Query', description: 'Ask questions about your documents' },
  { href: '/ingest', label: 'Ingest', description: 'Upload PDFs for processing' },
];

/**
 * Navbar component with navigation links
 *
 * Displays links to Query and Ingest pages with active state indication.
 */
export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mx-auto max-w-5xl px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo / App name */}
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold text-zinc-900 dark:text-zinc-50">
              RecRAG
            </span>
            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700 dark:bg-blue-900 dark:text-blue-300">
              Beta
            </span>
          </div>

          {/* Navigation links */}
          <div className="flex gap-1">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50'
                      : 'text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-50'
                  }`}
                  title={link.description}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
