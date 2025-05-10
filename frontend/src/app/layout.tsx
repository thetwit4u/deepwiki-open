import "./globals.css";
import type { Metadata } from "next";
import AppShell from "@/components/layout/AppShell";

export const metadata: Metadata = {
  title: "DeepWiki",
  description: "AI-powered codebase wiki and documentation explorer.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="font-sans">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
