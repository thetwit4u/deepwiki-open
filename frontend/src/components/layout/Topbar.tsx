import { UserCircle } from "lucide-react";

export default function Topbar() {
  return (
    <header className="flex items-center justify-between bg-white border-b border-gray-200 h-20 px-8">
      {/* Nike Logo (black) and Workspace Title */}
      <div className="flex items-center gap-4">
        <img src="/nike-logo-black.svg" width={32} height={32} alt="Nike logo (black)" />
        <span className="text-xl font-bold text-black">DeepWiki Workspace</span>
      </div>
      {/* User Avatar */}
      <div className="flex items-center gap-2">
        <UserCircle size={32} className="text-gray-400" />
        <span className="text-sm text-gray-700 font-medium">WW</span>
      </div>
    </header>
  );
} 