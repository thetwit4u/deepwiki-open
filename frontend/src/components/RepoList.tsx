import { Button } from "@/components/ui/button";

interface Repo {
  id: string;
  name: string;
  path: string;
}

interface RepoListProps {
  repos: Repo[];
  repoSearch: string;
  setRepoSearch: (s: string) => void;
  filteredRepos: Repo[];
  onSelectRepo: (repo: Repo) => void;
}

export default function RepoList({ repos, repoSearch, setRepoSearch, filteredRepos, onSelectRepo }: RepoListProps) {
  return (
    <>
      <div className="mb-4 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
        <div className="font-semibold text-lg">Existing Repositories</div>
        <input
          type="text"
          className="border rounded px-3 py-2 text-sm w-full sm:w-64"
          placeholder="Search repositories..."
          value={repoSearch}
          onChange={(e) => setRepoSearch(e.target.value)}
        />
      </div>
      <div className="max-h-[400px] overflow-y-auto border rounded bg-white shadow-sm">
        {filteredRepos.length === 0 ? (
          <div className="p-8 text-gray-400 text-center">No repositories found.</div>
        ) : (
          <div className="divide-y">
            {filteredRepos.map((repo) => (
              <div
                key={repo.id}
                className="flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition"
              >
                <div>
                  <div className="font-semibold">{repo.name}</div>
                  <div className="text-xs text-gray-500">{repo.path}</div>
                </div>
                <Button onClick={() => onSelectRepo(repo)} size="sm">
                  Select
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
} 