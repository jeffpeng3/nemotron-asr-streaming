import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "example",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      "@jeffpeng3/nemotron-asr-core": resolve(__dirname, "src/index.js"),
    },
  },
  worker: {
    format: "es",
  },
});
