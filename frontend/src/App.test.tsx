import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import App from "./App";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("application shell", () => {
  it("renders the Phase 02 shell without fake benchmark values", () => {
    renderAt("/");
    expect(screen.getByRole("heading", { name: /reproducible incident-diagnosis research/i })).toBeInTheDocument();
    expect(screen.getByText(/engineering foundation only/i)).toBeInTheDocument();
  });

  it("renders a route fallback for unknown paths", () => {
    renderAt("/does-not-exist");
    expect(screen.getByRole("heading", { name: /route not found/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /return home/i })).toHaveAttribute("href", "/");
  });
});
