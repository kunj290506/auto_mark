import React, { useState } from 'react'
import './Header.css'

const pageInfo = {
    upload: {
        title: 'Upload Images',
        description: 'Upload a zip file containing images for annotation'
    },
    configure: {
        title: 'Configure Detection',
        description: 'Define objects to detect and adjust settings'
    },
    processing: {
        title: 'Processing',
        description: 'Automated annotation in progress'
    },
    viewer: {
        title: 'Review Annotations',
        description: 'Review and edit detected objects'
    },
    export: {
        title: 'Export Dataset',
        description: 'Download your annotated dataset'
    }
}

function Header({ session, currentPage }) {
    const info = pageInfo[currentPage] || pageInfo.upload

    // UI 5: Dark mode toggle
    const [dark, setDark] = useState(true)
    const toggleTheme = () => {
        setDark(d => !d)
        document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark')
    }

    return (
        <header className="header">
            <div className="header-content">
                <div className="header-info">
                    <h1 className="header-title">{info.title}</h1>
                    <p className="header-description">{info.description}</p>
                </div>

                <div className="header-actions">
                    {/* UI 5: Theme toggle */}
                    <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
                        {dark ? (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
                                <circle cx="12" cy="12" r="5" />
                                <line x1="12" y1="1" x2="12" y2="3" />
                                <line x1="12" y1="21" x2="12" y2="23" />
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                                <line x1="1" y1="12" x2="3" y2="12" />
                                <line x1="21" y1="12" x2="23" y2="12" />
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                            </svg>
                        ) : (
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                            </svg>
                        )}
                    </button>

                    {session && (
                        <div className="header-session">
                            <div className="session-stat">
                                <span className="stat-value">{session.image_count}</span>
                                <span className="stat-label">Images</span>
                            </div>
                            <div className="session-divider" />
                            <div className="session-id">
                                <span className="id-label">Session</span>
                                <span className="id-value">{session.session_id?.slice(0, 8)}...</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    )
}

export default Header
