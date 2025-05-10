import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Section {
  id: string;
  title: string;
}

interface WikiSidebarProps {
  sections: Section[];
  open?: boolean;
  selectedSection?: string | null;
  onSelectSection?: (id: string) => void;
}

export default function WikiSidebar({ sections, open: initialOpen = true, selectedSection, onSelectSection }: WikiSidebarProps) {
  const [open, setOpen] = useState(initialOpen);

  return (
    <aside className={`transition-all duration-200 ${open ? 'w-64' : 'w-10'} bg-white border-r border-gray-200 h-full flex flex-col`}>
      <button
        className="self-end m-2 p-1 rounded hover:bg-gray-100"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Collapse wiki sidebar" : "Expand wiki sidebar"}
      >
        {open ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
      </button>
      {open && (
        <nav className="flex-1 overflow-y-auto px-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Wiki Sections</h3>
          <ul className="space-y-1">
            {sections.map((section) => (
              <li key={section.id}>
                <button
                  className={`w-full text-left px-3 py-2 rounded text-sm font-medium ${selectedSection === section.id ? 'bg-gray-200 font-bold' : 'hover:bg-gray-100'}`}
                  onClick={() => onSelectSection && onSelectSection(section.id)}
                >
                  {section.title}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </aside>
  );
} 