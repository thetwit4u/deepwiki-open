"use client";
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import WikiSidebar from "@/components/layout/WikiSidebar";
import SectionContent from "@/components/SectionContent";
import AddRepoCard from "@/components/AddRepoCard";
import RepoList from "@/components/RepoList";
import RepoDetailsCard from "@/components/RepoDetailsCard";
import ChatPanel from "@/components/ChatPanel";
import { useRouter, useSearchParams } from "next/navigation";

interface Repo {
  id: string;
  name: string;
  path: string;
  status?: string;
  has_structure?: boolean;
  wiki_path?: string;
}

interface Section {
  id: string;
  title: string;
}

export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [repos, setRepos] = useState<Repo[]>([]);
  const [selected, setSelected] = useState<Repo | null>(null);
  const [loading, setLoading] = useState(true);
  const [sections, setSections] = useState<Section[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [sectionContent, setSectionContent] = useState<string>("");
  const [sectionMetadata, setSectionMetadata] = useState<any>({});
  const [contentLoading, setContentLoading] = useState(false);
  const [repoSearch, setRepoSearch] = useState("");

  const fetchRepos = () => {
    setLoading(true);
    fetch("/api/repos")
      .then((res) => res.json())
      .then((data) => {
        setRepos(data.repos || []);
        setLoading(false);
        
        // Check if there's a repo ID in the URL and select it if it exists
        const repoIdFromUrl = searchParams.get('repo');
        if (repoIdFromUrl) {
          const repoFromUrl = (data.repos || []).find((r: Repo) => r.id === repoIdFromUrl);
          if (repoFromUrl) {
            setSelected(repoFromUrl);
          }
        }
      });
  };

  useEffect(() => {
    fetchRepos();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle repo selection and update URL
  const handleSelectRepo = (repo: Repo) => {
    setSelected(repo);
    
    // Update the URL with the repo ID
    const params = new URLSearchParams();
    params.set('repo', repo.id);
    router.push(`/?${params.toString()}`);
  };

  // Handle deselect repo and remove from URL
  const handleDeselectRepo = () => {
    setSelected(null);
    router.push('/');
  };

  // Handle section selection and update URL
  const handleSelectSection = (sectionId: string) => {
    setSelectedSection(sectionId);
    
    // Update the URL with both repo and section IDs
    const params = new URLSearchParams();
    if (selected) {
      params.set('repo', selected.id);
    }
    params.set('section', sectionId);
    router.push(`/?${params.toString()}`);
  };

  useEffect(() => {
    if (selected) {
      fetch("/api/wiki-structure?repo=" + encodeURIComponent(selected.id))
        .then((res) => res.json())
        .then((data) => {
          setSections(data.sections || []);
          
          // Check if there's a section ID in the URL
          const sectionIdFromUrl = searchParams.get('section');
          if (sectionIdFromUrl && data.sections && data.sections.find((s: Section) => s.id === sectionIdFromUrl)) {
            setSelectedSection(sectionIdFromUrl);
          }
          // Otherwise default to first section
          else if (data.sections && data.sections.length > 0) {
            handleSelectSection(data.sections[0].id);
          }
        });
    } else {
      setSections([]);
      setSelectedSection(null);
      setSectionContent("");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  useEffect(() => {
    if (selected && selectedSection) {
      setContentLoading(true);
      const apiUrl = `/api/section-content?repo=${encodeURIComponent(
        selected.id
      )}&section=${encodeURIComponent(selectedSection)}`;
      
      fetch(apiUrl)
        .then((res) => res.json())
        .then((data) => {
          setSectionContent(data.content || "");
          setSectionMetadata(data.metadata || {});
        })
        .catch(err => {
          console.error(`Error fetching content:`, err);
        })
        .finally(() => setContentLoading(false));
    } else {
      setSectionContent("");
      setSectionMetadata({});
    }
  }, [selected, selectedSection]);

  // Filter repos by search
  const filteredRepos = repoSearch.trim()
    ? repos.filter((r) =>
        r.name.toLowerCase().includes(repoSearch.toLowerCase()) ||
        r.path.toLowerCase().includes(repoSearch.toLowerCase())
      )
    : repos;

  if (loading) return <div className="p-8">Loading repositories...</div>;

  if (!selected) {
  return (
      <div className="p-8 max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold mb-2">Repository Management</h2>
        <div className="mb-8">
          <AddRepoCard onRepoAdded={fetchRepos} />
        </div>
        <RepoList
          repos={repos}
          repoSearch={repoSearch}
          setRepoSearch={setRepoSearch}
          filteredRepos={filteredRepos}
          onSelectRepo={handleSelectRepo}
        />
      </div>
    );
  }

  // Main content with WikiSidebar and SectionContent
  return (
    <div className="flex h-full min-h-[600px]">
      <WikiSidebar
        sections={sections}
        open={sidebarOpen}
        selectedSection={selectedSection}
        onSelectSection={handleSelectSection}
      />
      <div className="flex-1 pl-3 flex flex-col">
        <div className="flex justify-between items-center mb-2 mt-1">
          <div>
            <h2 className="text-lg font-semibold">{selected.name}</h2>
            <div className="text-xs text-gray-500">{selected.path}</div>
          </div>
          <Button 
            onClick={handleDeselectRepo} 
            size="sm" 
            variant="outline"
            className="text-xs h-7"
          >
            Choose another repo
          </Button>
        </div>
        
        <div className="flex flex-col space-y-3 flex-grow">
          {contentLoading ? (
            <div className="p-4 text-gray-500 text-center">Loading section content...</div>
          ) : (
            <>
              <SectionContent content={sectionContent} metadata={sectionMetadata} />
            </>
          )}
          
          <div className="mt-1">
            <ChatPanel repoId={selected.id} />
          </div>
        </div>
      </div>
    </div>
  );
}
