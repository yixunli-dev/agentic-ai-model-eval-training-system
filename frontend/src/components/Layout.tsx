import type { ReactNode } from "react";

import { navigateTo } from "../navigation";

type LayoutProps = {
  currentPath: string;
  children: ReactNode;
};

export default function Layout({ currentPath, children }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-mark">AE</div>
          <h1>Agentic Eval</h1>
          <p>Model training, evaluation, and workflow runs.</p>
        </div>
        <nav className="nav-links">
          <button className={currentPath === "/" ? "active" : ""} onClick={() => navigateTo("/")}>
            Dashboard
          </button>
          <button className={currentPath === "/new" ? "active" : ""} onClick={() => navigateTo("/new")}>
            New Run
          </button>
        </nav>
      </aside>
      <main className="main-content">{children}</main>
    </div>
  );
}
