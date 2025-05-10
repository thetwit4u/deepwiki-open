import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';

// Simple in-memory job store (not for production)
const globalAny = global as any;
if (!globalAny.repoJobs) globalAny.repoJobs = {};

const REPOS_DIR = path.resolve(process.cwd(), '../../repos');

function isGitUrl(url: string) {
  return /^(https?:\/\/|git@)/.test(url);
}

function copyDirSync(src: string, dest: string) {
  if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

export async function POST(req: NextRequest) {
  const { repoUrl } = await req.json();
  if (!repoUrl) {
    return NextResponse.json({ error: 'Missing repoUrl' }, { status: 400 });
  }
  // Generate a simple job ID
  const jobId = Math.random().toString(36).slice(2, 10);
  globalAny.repoJobs[jobId] = {
    status: 'queued',
    progress: 0,
    repoUrl,
    started: Date.now(),
    log: ['Queued'],
  };
  // Determine repo name
  let repoName = '';
  if (isGitUrl(repoUrl)) {
    repoName = repoUrl.split('/').pop()?.replace(/\.git$/, '') || 'repo';
  } else {
    repoName = path.basename(repoUrl);
  }
  const destDir = path.join(REPOS_DIR, repoName);
  // If already exists, skip clone/copy
  if (!fs.existsSync(destDir)) {
    try {
      if (isGitUrl(repoUrl)) {
        globalAny.repoJobs[jobId].status = 'cloning';
        globalAny.repoJobs[jobId].progress = 5;
        globalAny.repoJobs[jobId].log.push('Cloning repository...');
        await new Promise<void>((resolve, reject) => {
          exec(`git clone --depth 1 ${repoUrl} "${destDir}"`, (err) => {
            if (err) reject(err);
            else resolve();
          });
        });
      } else {
        globalAny.repoJobs[jobId].status = 'copying';
        globalAny.repoJobs[jobId].progress = 5;
        globalAny.repoJobs[jobId].log.push('Copying repository...');
        copyDirSync(repoUrl, destDir);
      }
    } catch (err) {
      globalAny.repoJobs[jobId].status = 'error';
      globalAny.repoJobs[jobId].progress = 0;
      globalAny.repoJobs[jobId].log.push('Failed to clone/copy repo');
      return NextResponse.json({ error: 'Failed to clone/copy repo', details: String(err) }, { status: 500 });
    }
  }
  // Simulate async pipeline
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'scanning';
    globalAny.repoJobs[jobId].progress = 20;
    globalAny.repoJobs[jobId].log.push('Scanning files...');
  }, 1000);
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'splitting';
    globalAny.repoJobs[jobId].progress = 35;
    globalAny.repoJobs[jobId].log.push('Splitting files into chunks...');
  }, 2000);
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'embedding';
    globalAny.repoJobs[jobId].progress = 55;
    globalAny.repoJobs[jobId].log.push('Embedding chunks...');
  }, 3500);
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'indexing';
    globalAny.repoJobs[jobId].progress = 75;
    globalAny.repoJobs[jobId].log.push('Indexing vectors...');
  }, 5000);
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'wiki-structure';
    globalAny.repoJobs[jobId].progress = 90;
    globalAny.repoJobs[jobId].log.push('Generating wiki structure...');
    // Store a mock wiki structure
    globalAny.repoJobs[jobId].wikiStructure = {
      sections: [
        { id: 'intro', title: 'Introduction' },
        { id: 'setup', title: 'Setup' },
        { id: 'usage', title: 'Usage' },
        { id: 'architecture', title: 'Architecture' },
        { id: 'api', title: 'API Reference' },
        { id: 'faq', title: 'FAQ' },
      ],
    };
  }, 6500);
  setTimeout(() => {
    globalAny.repoJobs[jobId].status = 'done';
    globalAny.repoJobs[jobId].progress = 100;
    globalAny.repoJobs[jobId].log.push('Done!');
  }, 8000);
  return NextResponse.json({ jobId });
} 