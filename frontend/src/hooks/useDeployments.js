import { useCallback, useEffect, useState } from "react";
import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8080", timeout: 30000 });

export function useDeployments() {
  const [items, setItems] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const { data } = await API.get("/api/deployments");
      setItems(Array.isArray(data) ? data : []);
      setError(null);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { items, error, loading, refresh };
}
