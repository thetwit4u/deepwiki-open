import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const REPOS_DIR = path.resolve(process.cwd(), '../../wiki-data/repos');

type Repo = { id: string; name: string; path: string };

export async function GET() {
  let repos: Repo[] = [];
  try {
    if (fs.existsSync(REPOS_DIR)) {
      const entries = fs.readdirSync(REPOS_DIR, { withFileTypes: true });
      repos = entries
        .filter((entry) => entry.isDirectory())
        .map((entry) => ({
          id: entry.name,
          name: entry.name,
          path: path.join('/repos', entry.name),
        }));
    }
  } catch (err) {
    // fallback to empty list
  }
  return NextResponse.json({ repos });
} 