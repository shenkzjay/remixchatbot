import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("rag", "./routes/rag.tsx"),
  route("lang", "./routes/langgraph.tsx"),
] satisfies RouteConfig;
