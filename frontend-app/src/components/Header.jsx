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
        description: 'Annotating your images with AI'
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

    return (
        <header className="header">
            <div className="header-content">
                <div className="header-info">
                    <h1 className="header-title">{info.title}</h1>
                    <p className="header-description">{info.description}</p>
                </div>

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
        </header>
    )
}

export default Header
