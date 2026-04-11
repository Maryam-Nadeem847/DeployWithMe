import ActiveDeployments from "../components/ActiveDeployments.jsx";
import { useDeployments } from "../hooks/useDeployments.js";

export default function DashboardPage() {
  const { items, loading, refresh } = useDeployments();

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <section className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Deployment Dashboard</h1>
        <p className="mt-2 text-slate-600">
          Monitor, test, and stop all active model deployments.
        </p>
      </section>
      <ActiveDeployments items={items} loading={loading} onRefresh={refresh} />
    </div>
  );
}
