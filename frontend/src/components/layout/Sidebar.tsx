import { Home, Book, Search, Settings } from "lucide-react";

const navItems = [
  { icon: <Home size={24} />, label: "Home" },
  { icon: <Book size={24} />, label: "Wiki" },
  { icon: <Search size={24} />, label: "Search" },
  { icon: <Settings size={24} />, label: "Settings" },
];

export default function Sidebar() {
  return (
    <aside className="flex flex-col items-center bg-black text-white w-20 min-h-screen py-6">
      {/* Nike Logo (white) */}
      <div className="mb-10">
        <img src="/nike-logo-white.png" width={32} height={32} alt="Nike logo (white)" />
      </div>
      {/* Nav Items */}
      <nav className="flex flex-col gap-8 mt-4">
        {navItems.map((item, idx) => (
          <button
            key={item.label}
            className="flex flex-col items-center focus:outline-none opacity-80 hover:opacity-100"
            aria-label={item.label}
          >
            {item.icon}
            <span className="sr-only">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
} 