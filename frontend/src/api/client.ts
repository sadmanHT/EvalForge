export interface HealthResponse {
  status: string;
  service: string;
}

const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchHealth(baseUrl = defaultBaseUrl): Promise<HealthResponse> {
  const response = await fetch(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error(`Health request failed with status ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}
