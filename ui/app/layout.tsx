import type { Metadata } from "next";
import "./globals.css";
import { FlowBreadcrumb } from "@/components/FlowBreadcrumb";

export const metadata: Metadata = {
  title: "LLM Inference Capacity Planner",
  description: "Size, cost, and validate LLM inference deployments",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 min-h-screen">
        <header className="bg-slate-900 px-6 h-13 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-400 to-indigo-600 flex items-center justify-center text-sm flex-shrink-0">
              ⚡
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-100 leading-tight group-hover:text-white">
                LLM Inference Planner
              </div>
              <div className="text-xs text-slate-500 leading-tight">
                GPU Sizing · Roofline Model
              </div>
            </div>
          </a>
          <nav className="flex items-center gap-5 text-sm text-slate-400">
            <a href="/" className="hover:text-slate-100 transition-colors">New Scenario</a>
            <a
              href="https://github.com/amitgambhir/llm-inference-planner/blob/main/README.md"
              target="_blank"
              rel="noreferrer"
              className="hover:text-slate-100 transition-colors"
            >
              Docs
            </a>
            <span className="bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded px-2 py-0.5">
              v1.0.0
            </span>
          </nav>
        </header>
        <FlowBreadcrumb />
        <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
