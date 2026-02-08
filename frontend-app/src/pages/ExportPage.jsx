import { useState } from 'react'
import './ExportPage.css'

const API_URL = ''

function ExportPage({ session, annotations, config, onBack, onNewProject }) {
    const [selectedFormat, setSelectedFormat] = useState(config.exportFormat)
    const [exporting, setExporting] = useState(false)
    const [exportComplete, setExportComplete] = useState(false)

    const formats = [
        {
            id: 'coco',
            name: 'COCO JSON',
            icon: '📦',
            description: 'Industry standard format for object detection. Single JSON file with all annotations.',
            features: ['Single file', 'Segmentation support', 'Category hierarchy']
        },
        {
            id: 'yolo',
            name: 'YOLO Format',
            icon: '⚡',
            description: 'Optimized for YOLO models. Separate label files per image with normalized coordinates.',
            features: ['Fast training', 'Lightweight', 'Easy integration']
        },
        {
            id: 'voc',
            name: 'Pascal VOC',
            icon: '📄',
            description: 'XML-based format compatible with many detection frameworks.',
            features: ['XML format', 'Verbose metadata', 'Wide compatibility']
        },
        {
            id: 'roboflow',
            name: 'Roboflow',
            icon: '🚀',
            description: 'Ready to upload to Roboflow platform for augmentation and training.',
            features: ['Platform ready', 'Augmentation', 'Version control']
        }
    ]

    // Stats
    const totalImages = annotations?.images?.length || 0
    const totalDetections = Object.values(annotations?.annotations || {}).reduce(
        (sum, ann) => sum + (ann.boxes?.length || 0), 0
    )
    const classes = [...new Set(
        Object.values(annotations?.annotations || {}).flatMap(ann => ann.labels || [])
    )]

    const handleExport = async () => {
        setExporting(true)

        try {
            const response = await fetch(
                `${API_URL}/api/export/${session.session_id}?format=${selectedFormat}`,
                { method: 'POST' }
            )

            if (!response.ok) {
                throw new Error('Export failed')
            }

            // Download the file
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `annotations_${session.session_id.slice(0, 8)}_${selectedFormat}.zip`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)

            setExportComplete(true)
        } catch (error) {
            console.error('Export error:', error)
            alert('Export failed. Please try again.')
        } finally {
            setExporting(false)
        }
    }

    return (
        <div className="export-page">
            {!exportComplete ? (
                <>
                    {/* Stats Summary */}
                    <div className="export-stats glass-card animate-slideUp">
                        <div className="stat-item">
                            <span className="stat-icon">📁</span>
                            <div className="stat-content">
                                <span className="stat-value">{totalImages}</span>
                                <span className="stat-label">Images</span>
                            </div>
                        </div>
                        <div className="stat-divider" />
                        <div className="stat-item">
                            <span className="stat-icon">🎯</span>
                            <div className="stat-content">
                                <span className="stat-value">{totalDetections}</span>
                                <span className="stat-label">Detections</span>
                            </div>
                        </div>
                        <div className="stat-divider" />
                        <div className="stat-item">
                            <span className="stat-icon">🏷️</span>
                            <div className="stat-content">
                                <span className="stat-value">{classes.length}</span>
                                <span className="stat-label">Classes</span>
                            </div>
                        </div>
                    </div>

                    {/* Classes */}
                    <div className="classes-section animate-slideUp delay-100">
                        <h4>Detected Classes</h4>
                        <div className="classes-grid">
                            {classes.map((cls, i) => (
                                <span key={i} className="class-badge">
                                    {cls}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Format Selection */}
                    <div className="format-section animate-slideUp delay-200">
                        <h3>Select Export Format</h3>
                        <div className="formats-grid">
                            {formats.map((format) => (
                                <label
                                    key={format.id}
                                    className={`format-card glass-card ${selectedFormat === format.id ? 'selected' : ''}`}
                                >
                                    <input
                                        type="radio"
                                        name="format"
                                        value={format.id}
                                        checked={selectedFormat === format.id}
                                        onChange={(e) => setSelectedFormat(e.target.value)}
                                    />
                                    <div className="format-header">
                                        <span className="format-icon">{format.icon}</span>
                                        <h4 className="format-name">{format.name}</h4>
                                    </div>
                                    <p className="format-description">{format.description}</p>
                                    <ul className="format-features">
                                        {format.features.map((feature, i) => (
                                            <li key={i}>
                                                <svg viewBox="0 0 16 16" fill="currentColor">
                                                    <path d="M13.485 4.515a1 1 0 0 0-1.414 0L6.5 10.086 3.929 7.515a1 1 0 1 0-1.414 1.414l3.285 3.285a1 1 0 0 0 1.414 0l6.271-6.271a1 1 0 0 0 0-1.414z" />
                                                </svg>
                                                {feature}
                                            </li>
                                        ))}
                                    </ul>
                                    <div className="format-check">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                            <polyline points="20,6 9,17 4,12" />
                                        </svg>
                                    </div>
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="export-actions animate-slideUp delay-300">
                        <button className="btn btn-secondary" onClick={onBack}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="19" y1="12" x2="5" y2="12" />
                                <polyline points="12,19 5,12 12,5" />
                            </svg>
                            Back to Viewer
                        </button>
                        <button
                            className="btn btn-primary btn-lg"
                            onClick={handleExport}
                            disabled={exporting}
                        >
                            {exporting ? (
                                <>
                                    <svg className="animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                    </svg>
                                    Exporting...
                                </>
                            ) : (
                                <>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="7,10 12,15 17,10" />
                                        <line x1="12" y1="15" x2="12" y2="3" />
                                    </svg>
                                    Download Dataset
                                </>
                            )}
                        </button>
                    </div>
                </>
            ) : (
                /* Success State */
                <div className="export-success animate-slideUp">
                    <div className="success-icon">
                        <svg viewBox="0 0 64 64" fill="none">
                            <circle cx="32" cy="32" r="30" stroke="url(#successGradient)" strokeWidth="4" />
                            <path d="M20 32l8 8 16-16" stroke="url(#successGradient)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                            <defs>
                                <linearGradient id="successGradient" x1="0" y1="0" x2="64" y2="64">
                                    <stop stopColor="#10b981" />
                                    <stop offset="1" stopColor="#06b6d4" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    <h2>Export Complete!</h2>
                    <p>Your annotated dataset has been downloaded successfully.</p>

                    <div className="success-stats glass-card">
                        <div className="success-stat">
                            <span className="value">{totalImages}</span>
                            <span className="label">Images</span>
                        </div>
                        <div className="success-stat">
                            <span className="value">{totalDetections}</span>
                            <span className="label">Annotations</span>
                        </div>
                        <div className="success-stat">
                            <span className="value">{selectedFormat.toUpperCase()}</span>
                            <span className="label">Format</span>
                        </div>
                    </div>

                    <div className="success-actions">
                        <button className="btn btn-secondary" onClick={() => setExportComplete(false)}>
                            Export Another Format
                        </button>
                        <button className="btn btn-primary" onClick={onNewProject}>
                            Start New Project
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default ExportPage
