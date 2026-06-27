export default function ReportViewer({ report }: { report: string }) {
  if (!report) {
    return <div className="empty-state">No report found.</div>;
  }

  return <pre className="report-viewer">{report}</pre>;
}
