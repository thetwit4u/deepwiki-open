import { Button } from "@/components/ui/button";

interface Repo {
  id: string;
  name: string;
  path: string;
  status?: string;
  has_structure?: boolean;
  wiki_path?: string;
}

interface RepoListProps {
  repos: Repo[];
  repoSearch: string;
  setRepoSearch: (s: string) => void;
  filteredRepos: Repo[];
  onSelectRepo: (repo: Repo) => void;
}

export default function RepoList({ repos, repoSearch, setRepoSearch, filteredRepos, onSelectRepo }: RepoListProps) {
  // Helper function to get status badge color
  const getStatusColor = (status?: string) => {
    switch (status) {
      case "done":
        return "bg-green-100 text-green-800";
      case "error":
        return "bg-red-100 text-red-800";
      case "generating section content":
        return "bg-blue-100 text-blue-800";
      case "generating wiki structure":
        return "bg-indigo-100 text-indigo-800";
      case "indexing":
        return "bg-purple-100 text-purple-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <>
      <div className="mb-4 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
        <div className="font-semibold text-lg">Generated Wikis</div>
        <input
          type="text"
          className="border rounded px-3 py-2 text-sm w-full sm:w-64"
          placeholder="Search wikis..."
          value={repoSearch}
          onChange={(e) => setRepoSearch(e.target.value)}
        />
      </div>
      <div className="max-h-[400px] overflow-y-auto border rounded bg-white shadow-sm">
        {filteredRepos.length === 0 ? (
          <div className="p-8 text-gray-400 text-center">No wikis found.</div>
        ) : (
          <div className="divide-y">
            {filteredRepos.map((repo) => (
              <div
                key={repo.id}
                className="flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition"
              >
                <div>
                  <div className="font-semibold">{repo.name}</div>
                  <div className="text-xs text-gray-500">{repo.path || repo.wiki_path || 'No path'}</div>
                  {repo.status && (
                    <div className={`text-xs mt-1 px-2 py-0.5 rounded-full inline-block ${getStatusColor(repo.status)}`}>
                      {repo.status}
                    </div>
                  )}
                </div>
                <Button 
                  onClick={() => onSelectRepo(repo)} 
                  size="sm"
                  disabled={repo.status !== "done" && !repo.has_structure}>
                  {repo.status === "done" || repo.has_structure ? "View Wiki" : "In Progress"}
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
} 