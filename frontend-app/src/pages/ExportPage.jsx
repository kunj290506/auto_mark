import React, { useState } from 'react'
import { API_URL } from '../config'
import SkeletonLoader from '../components/SkeletonLoader'
import './ExportPage.css'

// UI 7: Format comparison table
const FORMAT_COMPARE = [
    { feature: 'Single file output', coco: true, yolo: false, voc: false, roboflow: true },
    { feature: 'Segmentation masks', coco: true, yolo: false, voc: false, roboflow: true },
    { feature: 'Polygon annotations', coco: true, yolo: false, voc: false, roboflow: true },
    { feature: 'Per-image files', coco: false, yolo: true, voc: true, roboflow: false },
    { feature: 'Training ready (YAML)', coco: false, yolo: true, voc: false, roboflow: false },
    { feature: 'Ultralytics compatible', coco: false, yolo: true, voc: false, roboflow: true },
    { feature: 'Standard XML format', coco: false, yolo: false, voc: true, roboflow: false }
]

function ExportPage({ session, annotations, config, onBack, onNewProject }) {
    const [selectedFormat, setSelectedFormat] = useState(config?.exportFormat || 'coco')
    const [exporting, setExporting] = useState(false)
    const [exported, setExported] = useState(false)
    const [error, setError] = useState(null)
    const [showCompare, setShowCompare] = useState(false)

    const stats = annotations ? {
        total_images: Object.keys(annotations.annotations || {}).length,
        total_detections: Object.values(annotations.annotations || {}).reduce(
            (sum, ann) => sum + (ann.boxes?.length || 0), 0
        ),
        classes: [...new Set(
            Object.values(annotations.annotations || {}).flatMap(ann => ann.labels || [])
        )]
    } : null

    const formats = [
        { id: 'coco', name: 'COCO JSON', desc: 'Industry standard. Single JSON file with bounding boxes and segmentation polygons.', icon: 'C' },
        { id: 'yolo', name: 'YOLO', desc: 'Training-ready format. Per-image .txt files with data.yaml configuration.', icon: 'Y' },
        { id: 'voc', name: 'Pascal VOC', desc: 'XML-based format. Individual XML files for each image with bounding box coordinates.', icon: 'V' },
        { id: 'roboflow', name: 'Roboflow', desc: 'Roboflow platform compatible. COCO format with additional metadata.', icon: 'R' }
    ]

    const handleExport = async () => {
        setExporting(true)
        setError(null)

        try {
            const response = await fetch(`${API_URL}/api/export/${session.session_id}?format=${selectedFormat}`, {
                method: 'POST'
            })

            if (!response.ok) {
                throw new Error('Export failed')
            }

            // Download the zip file
            const blob = await response.blob()
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `annotations_${session.session_id}_${selectedFormat}.zip`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)

            setExported(true)
        } catch (err) {
            setError(err.message)
        } finally {
            setExporting(false)
        }
    }

    return (
        <div className="export-page">
            {/* Dataset Summary */}
            <div className="export-summary glass-card animate-slideUp">
                <h3>Dataset Summary</h3>
                {stats ? (
                    <div className="summary-grid">
                        <div className="summary-item">
                            <span className="summary-value">{stats.total_images}</span>
                            <span className="summary-label">Images</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-value">{stats.total_detections}</span>
                            <span className="summary-label">Detections</span>
                        </div>
                        <div className="summary-item">
                            <span className="summary-value">{stats.classes.length}</span>
                            <span className="summary-label">Classes</span>
                        </div>
                        <div className="summary-item classes-list">
                            {stats.classes.map((cls, i) => (
                                <span key={i} className="class-chip">{cls}</span>
                            ))}
                        </div>
                    </div>
                ) : (
                    /* UI 8: Skeleton loader */
                    <div className="summary-grid">
                        {[1, 2, 3].map(i => (
                            <div key={i} className="summary-item">
                                <SkeletonLoader width="80px" height="36px" borderRadius="8px" />
                                <SkeletonLoader width="60px" height="14px" borderRadius="4px" />
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Format Selection */}
            <div className="format-section animate-slideUp delay-100">
                <div className="format-header">
                    <h3>Export Format</h3>
                    <button className="btn btn-sm btn-ghost" onClick={() => setShowCompare(v => !v)}>
                        {showCompare ? 'Hide Comparison' : 'Compare Formats'}
                    </button>
                </div>

                {/* UI 7: Format comparison table */}
                {showCompare && (
                    <div className="format-compare glass-card animate-slideUp">
                        <table className="compare-table">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>COCO</th>
                                    <th>YOLO</th>
                                    <th>VOC</th>
                                    <th>Roboflow</th>
                                </tr>
                            </thead>
                            <tbody>
                                {FORMAT_COMPARE.map((row, i) => (
                                    <tr key={i}>
                                        <td>{row.feature}</td>
                                        <td className={row.coco ? 'yes' : 'no'}>{row.coco ? 'Yes' : '--'}</td>
                                        <td className={row.yolo ? 'yes' : 'no'}>{row.yolo ? 'Yes' : '--'}</td>
                                        <td className={row.voc ? 'yes' : 'no'}>{row.voc ? 'Yes' : '--'}</td>
                                        <td className={row.roboflow ? 'yes' : 'no'}>{row.roboflow ? 'Yes' : '--'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                <div className="format-cards">
                    {formats.map(fmt => (
                        <div
                            key={fmt.id}
                            className={`format-card glass-card glass-card-hover ${selectedFormat === fmt.id ? 'selected' : ''}`}
                            onClick={() => setSelectedFormat(fmt.id)}
                        >
                            <div className="format-icon">{fmt.icon}</div>
                            <h4>{fmt.name}</h4>
                            <p>{fmt.desc}</p>
                            {selectedFormat === fmt.id && (
                                <div className="format-selected-check">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" width="20" height="20">
                                        <polyline points="20,6 9,17 4,12" />
                                    </svg>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Actions */}
            <div className="export-actions animate-slideUp delay-200">
                {error && (
                    <div className="error-message">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        {error}
                    </div>
                )}

                {exported && (
                    <div className="success-message">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <polyline points="22,4 12,14.01 9,11.01" />
                        </svg>
                        Export downloaded successfully!
                    </div>
                )}

                <div className="action-buttons">
                    <button className="btn btn-secondary" onClick={onBack}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                            <polyline points="15,18 9,12 15,6" />
                        </svg>
                        Back to Viewer
                    </button>

                    <button
                        className="btn btn-primary btn-lg"
                        onClick={handleExport}
                        disabled={exporting}
                    >
                        {exporting ? (
                            <>Exporting...</>
                        ) : (
                            <>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="7,10 12,15 17,10" />
                                    <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                                Download {selectedFormat.toUpperCase()} Export
                            </>
                        )}
                    </button>

                    <button className="btn btn-outline" onClick={onNewProject}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                            <line x1="12" y1="5" x2="12" y2="19" />
                            <line x1="5" y1="12" x2="19" y2="12" />
                        </svg>
                        New Project
                    </button>
                </div>
            </div>
        </div>
    )
}

export default ExportPage
