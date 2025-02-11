import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [index("routes/rag.tsx"), route("rag", "./routes/home.tsx")] satisfies RouteConfig;
