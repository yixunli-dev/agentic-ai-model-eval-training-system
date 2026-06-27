import { useEffect, useState } from "react";
import Layout from "./components/Layout";
import DashboardPage from "./pages/DashboardPage";
import NewRunPage from "./pages/NewRunPage";
import RunDetailPage from "./pages/RunDetailPage";

function getCurrentPath() {
  return window.location.pathname;
}

export default function App() {
  const [path, setPath] = useState(getCurrentPath());

  useEffect(() => {
    const handlePathChange = () => setPath(getCurrentPath());
    window.addEventListener("popstate", handlePathChange);
    return () => window.removeEventListener("popstate", handlePathChange);
  }, []);

  let page = <DashboardPage />;
  if (path === "/new") {
    page = <NewRunPage />;
  } else if (path.startsWith("/runs/")) {
    page = <RunDetailPage runId={decodeURIComponent(path.replace("/runs/", ""))} />;
  }

  return <Layout currentPath={path}>{page}</Layout>;
}
