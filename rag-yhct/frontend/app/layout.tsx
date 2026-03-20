import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG YHCT Thesis Demo",
  description: "Hybrid RRF + Answerability Gate thesis demonstration",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
