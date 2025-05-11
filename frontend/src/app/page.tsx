"use client";
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import WikiSidebar from "@/components/layout/WikiSidebar";
import SectionContent from "@/components/SectionContent";
import AddRepoCard from "@/components/AddRepoCard";
import RepoList from "@/components/RepoList";
import RepoDetailsCard from "@/components/RepoDetailsCard";

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
  const [repos, setRepos] = useState<Repo[]>([]);
  const [selected, setSelected] = useState<Repo | null>(null);
  const [loading, setLoading] = useState(true);
  const [sections, setSections] = useState<Section[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [sectionContent, setSectionContent] = useState<string>("");
  const [contentLoading, setContentLoading] = useState(false);
  const [repoSearch, setRepoSearch] = useState("");

  const fetchRepos = () => {
    setLoading(true);
    fetch("/api/repos")
      .then((res) => res.json())
      .then((data) => {
        setRepos(data.repos || []);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchRepos();
  }, []);

  useEffect(() => {
    if (selected) {
      fetch("/api/wiki-structure?repo=" + encodeURIComponent(selected.id))
        .then((res) => res.json())
        .then((data) => {
          setSections(data.sections || []);
          // Default to first section
          if (data.sections && data.sections.length > 0) {
            setSelectedSection(data.sections[0].id);
          }
        });
    } else {
      setSections([]);
      setSelectedSection(null);
      setSectionContent("");
    }
  }, [selected]);

  useEffect(() => {
    if (selected && selectedSection) {
      setContentLoading(true);
      fetch(
        `/api/section-content?repo=${encodeURIComponent(
          selected.id
        )}&section=${encodeURIComponent(selectedSection)}`
      )
        .then((res) => res.json())
        .then((data) => setSectionContent(data.content || ""))
        .finally(() => setContentLoading(false));
    } else {
      setSectionContent("");
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
          onSelectRepo={setSelected}
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
        onSelectSection={setSelectedSection}
      />
      <RepoDetailsCard repo={selected} onDeselect={() => setSelected(null)}>
        {contentLoading ? (
          <div className="p-8 text-gray-500">Loading section content...</div>
        ) : (
          <SectionContent content={sectionContent} />
        )}
      </RepoDetailsCard>
    </div>
  );
}
