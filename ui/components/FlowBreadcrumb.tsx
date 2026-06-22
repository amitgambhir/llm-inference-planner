"use client";

import { usePathname } from "next/navigation";

const STEPS = [
  { label: "① Scenario",       href: "/" },
  { label: "② Estimate",       href: "/estimate" },
  { label: "③ Benchmark Plan", href: "/benchmark-plan" },
  { label: "④ Report",         href: "/report" },
];

function activeIndex(pathname: string): number {
  // longest matching prefix wins
  let best = 0;
  STEPS.forEach((s, i) => {
    if (s.href === "/" ? pathname === "/" : pathname.startsWith(s.href)) best = i;
  });
  return best;
}

export function FlowBreadcrumb() {
  const pathname = usePathname();
  const active = activeIndex(pathname);

  return (
    <div className="bg-slate-800 border-b border-slate-700 flex items-center px-6 h-8">
      {STEPS.map((step, i) => (
        <span key={step.href} className="flex items-center">
          {i > 0 && <span className="text-slate-600 text-xs mx-1">›</span>}
          <span
            className={[
              "text-xs font-medium px-2 h-8 flex items-center relative",
              i === active
                ? "text-indigo-300 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-indigo-500"
                : i < active
                ? "text-indigo-500 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-indigo-500 after:opacity-30"
                : "text-slate-500",
            ].join(" ")}
          >
            {step.label}
          </span>
        </span>
      ))}
    </div>
  );
}
