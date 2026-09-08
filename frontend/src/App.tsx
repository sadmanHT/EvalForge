import { Link, Route, Routes } from "react-router-dom";

function Shell() {
  return (
    <main className="shell">
      <header>
        <p className="eyebrow">EvalForge</p>
        <h1>Reproducible incident-diagnosis research</h1>
        <p>
          Phase 02 establishes the application shell and engineering foundation. Benchmark results are intentionally not
          rendered before real experiment evidence exists.
        </p>
      </header>
      <nav aria-label="Primary navigation">
        <Link to="/">Home</Link>
        <Link to="/status">Status</Link>
      </nav>
    </main>
  );
}

function StatusPage() {
  return (
    <section className="page">
      <h2>Foundation status</h2>
      <p>Backend, worker, storage, queue, frontend, and CI skeletons are wired for Phase 02.</p>
    </section>
  );
}

function NotFound() {
  return (
    <section className="page">
      <h2>Route not found</h2>
      <Link to="/">Return home</Link>
    </section>
  );
}

export default function App() {
  return (
    <>
      <Shell />
      <Routes>
        <Route path="/" element={<section className="page">Engineering foundation only.</section>} />
        <Route path="/status" element={<StatusPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
