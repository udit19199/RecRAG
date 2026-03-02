'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navLinks = [
  { href: '/', label: 'Chat', description: 'Ask questions about your documents' },
  { href: '/ingest', label: 'Documents', description: 'Upload and manage documents' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="bg-gradient-to-r from-zinc-950 via-zinc-900 to-zinc-950">
      <div className="mx-auto max-w-7xl px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo / App name */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-sky-500 shadow-md">
                <span className="text-sm font-bold text-white">R</span>
              </div>
              <div>
                <div className="text-lg font-semibold text-zinc-50">RecRAG</div>
                <div className="text-[11px] text-zinc-400">Retrieval + RAG</div>
              </div>
              <span className="rounded-full bg-blue-900/60 px-2 py-0.5 text-xs font-medium text-blue-300">
                Beta
              </span>
            </div>
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
                    ? 'bg-zinc-800 text-zinc-50'
                      : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-50'
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
