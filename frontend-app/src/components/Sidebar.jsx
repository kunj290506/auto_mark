import './Sidebar.css'

// Icons as SVG components
const Icons = {
    Logo: () => (
        <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="2" width="28" height="28" rx="6" fill="url(#logoGradient)" />
            <path d="M10 16L14 20L22 12" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <defs>
                <linearGradient id="logoGradient" x1="2" y1="2" x2="30" y2="30">
                    <stop stopColor="#6366f1" />
                    <stop offset="0.5" stopColor="#8b5cf6" />
                    <stop offset="1" stopColor="#06b6d4" />
                </linearGradient>
            </defs>
        </svg>
    ),
    Upload: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17,8 12,3 7,8" />
            <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
    ),
    Settings: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
    ),
    Play: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polygon points="10,8 16,12 10,16" fill="currentColor" />
        </svg>
    ),
    Eye: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
        </svg>
    ),
    Download: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7,10 12,15 17,10" />
            <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
    ),
    Check: () => (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20,6 9,17 4,12" />
        </svg>
    )
}

const navItems = [
    { id: 'upload', label: 'Upload', icon: Icons.Upload, step: 1 },
    { id: 'configure', label: 'Configure', icon: Icons.Settings, step: 2 },
    { id: 'processing', label: 'Process', icon: Icons.Play, step: 3 },
    { id: 'viewer', label: 'Review', icon: Icons.Eye, step: 4 },
    { id: 'export', label: 'Export', icon: Icons.Download, step: 5 }
]

function Sidebar({ currentPage, onNavigate, session }) {
    const currentStep = navItems.findIndex(item => item.id === currentPage) + 1

    const isStepAccessible = (step) => {
        if (step === 1) return true
        if (!session) return false
        return step <= currentStep + 1
    }

    const isStepCompleted = (step) => {
        return step < currentStep
    }

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <Icons.Logo />
                    <span className="logo-text">AutoMark</span>
                </div>
                <span className="logo-badge">AI</span>
            </div>

            <nav className="sidebar-nav">
                <div className="nav-label">Workflow</div>
                {navItems.map((item) => {
                    const Icon = item.icon
                    const accessible = isStepAccessible(item.step)
                    const completed = isStepCompleted(item.step)
                    const active = currentPage === item.id

                    return (
                        <button
                            key={item.id}
                            className={`nav-item ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${!accessible ? 'disabled' : ''}`}
                            onClick={() => accessible && onNavigate(item.id)}
                            disabled={!accessible}
                        >
                            <div className="nav-step">
                                {completed ? <Icons.Check /> : item.step}
                            </div>
                            <div className="nav-icon">
                                <Icon />
                            </div>
                            <span className="nav-label-text">{item.label}</span>
                            {active && <div className="nav-indicator" />}
                        </button>
                    )
                })}
            </nav>

            <div className="sidebar-footer">
                <div className="sidebar-info glass-card">
                    <div className="info-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 16v-4" />
                            <path d="M12 8h.01" />
                        </svg>
                    </div>
                    <div className="info-content">
                        <span className="info-title">Powered by</span>
                        <span className="info-text">Grounding DINO</span>
                    </div>
                </div>
            </div>
        </aside>
    )
}

export default Sidebar
