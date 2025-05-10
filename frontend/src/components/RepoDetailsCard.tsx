import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Repo {
  id: string;
  name: string;
  path: string;
}

interface RepoDetailsCardProps {
  repo: Repo;
  onDeselect: () => void;
  children: React.ReactNode;
}

export default function RepoDetailsCard({ repo, onDeselect, children }: RepoDetailsCardProps) {
  return (
    <div className="flex-1 pl-4">
      <h2 className="text-2xl font-bold mb-4">Repository Selected</h2>
      <Card className="p-4 mb-4">
        <div className="font-semibold">{repo.name}</div>
        <div className="text-sm text-gray-500">{repo.path}</div>
      </Card>
      <Button onClick={onDeselect} className="mb-6">
        Choose another repo
      </Button>
      {children}
    </div>
  );
} 