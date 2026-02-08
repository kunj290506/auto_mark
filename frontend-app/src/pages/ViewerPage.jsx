import { useState, useRef, useEffect } from 'react'
import './ViewerPage.css'

const API_URL = ''

function ViewerPage({ session, annotations, onExport, onBack }) {
    const [selectedImage, setSelectedImage] = useState(0)
    const [zoom, setZoom] = useState(1)
    const [showLabels, setShowLabels] = useState(true)
    const [selectedClass, setSelectedClass] = useState('all')
    const canvasRef = useRef(null)

    const images = annotations?.images || []
    const allAnnotations = annotations?.annotations || {}

    // Get unique classes
    const classes = [...new Set(
        Object.values(allAnnotations).flatMap(ann => ann.labels || [])
    )]

    // Current image data
    const currentImagePath = images[selectedImage]
    const currentAnnotation = allAnnotations[currentImagePath] || { boxes: [], labels: [], scores: [] }

    // Color map for classes
    const classColors = {
        default: '#6366f1',
        colors: ['#6366f1', '#06b6d4', '#10b981', '#f97316', '#ec4899', '#8b5cf6', '#f59e0b']
    }

    const getClassColor = (label) => {
        const index = classes.indexOf(label)
        return classColors.colors[index % classColors.colors.length] || classColors.default
    }

    // Draw annotations on canvas
    useEffect(() => {
        if (!canvasRef.current || !currentImagePath) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const img = new Image()
        img.crossOrigin = 'anonymous'

        img.onload = () => {
            canvas.width = img.width
            canvas.height = img.height

            // Draw image
            ctx.drawImage(img, 0, 0)

            // Get annotation data
            const boxes = currentAnnotation.boxes || []
            const labels = currentAnnotation.labels || []
            const scores = currentAnnotation.scores || []
            const segmentations = currentAnnotation.segmentations || []

            boxes.forEach((box, i) => {
                const label = labels[i] || 'unknown'
                const score = scores[i] || 0

                // Filter by class if selected
                if (selectedClass !== 'all' && label !== selectedClass) return

                const color = getClassColor(label)
                const x = box.x1 || (box.x * img.width)
                const y = box.y1 || (box.y * img.height)
                const w = box.x2 ? (box.x2 - box.x1) : (box.width * img.width)
                const h = box.y2 ? (box.y2 - box.y1) : (box.height * img.height)

                // Draw segmentation polygon if available
                const segmentation = segmentations[i]
                if (segmentation && segmentation.length > 0 && segmentation[0].length >= 6) {
                    const poly = segmentation[0]

                    // Draw filled polygon
                    ctx.beginPath()
                    ctx.moveTo(poly[0], poly[1])
                    for (let j = 2; j < poly.length; j += 2) {
                        ctx.lineTo(poly[j], poly[j + 1])
                    }
                    ctx.closePath()

                    // Fill with semi-transparent color
                    ctx.fillStyle = color + '30'
                    ctx.fill()

                    // Draw polygon outline
                    ctx.strokeStyle = color
                    ctx.lineWidth = 2
                    ctx.stroke()
                } else {
                    // Fallback to bounding box
                    ctx.strokeStyle = color
                    ctx.lineWidth = 3
                    ctx.strokeRect(x, y, w, h)
                    ctx.fillStyle = color + '20'
                    ctx.fillRect(x, y, w, h)
                }

                // Draw label
                if (showLabels) {
                    const labelText = `${label} ${(score * 100).toFixed(0)}%`
                    ctx.font = 'bold 14px Inter, sans-serif'
                    const textWidth = ctx.measureText(labelText).width

                    // Label background
                    ctx.fillStyle = color
                    ctx.fillRect(x, y - 24, textWidth + 12, 24)

                    // Label text
                    ctx.fillStyle = 'white'
                    ctx.fillText(labelText, x + 6, y - 7)
                }
            })
        }

        const imageName = currentImagePath.split(/[/\\]/).pop()
        img.src = `${API_URL}/uploads/${session?.session_id}/${imageName}`
    }, [selectedImage, currentAnnotation, showLabels, selectedClass, zoom])

    // Stats
    const totalDetections = Object.values(allAnnotations).reduce(
        (sum, ann) => sum + (ann.boxes?.length || 0), 0
    )
    const avgConfidence = Object.values(allAnnotations).reduce(
        (sum, ann) => sum + (ann.scores?.reduce((s, sc) => s + sc, 0) || 0), 0
    ) / (totalDetections || 1)

    return (
        <div className="viewer-page">
            {/* Toolbar */}
            <div className="viewer-toolbar glass-card">
                <div className="toolbar-left">
                    <button className="btn btn-ghost" onClick={onBack}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="19" y1="12" x2="5" y2="12" />
                            <polyline points="12,19 5,12 12,5" />
                        </svg>
                        Back
                    </button>
                </div>

                <div className="toolbar-center">
                    <button
                        className="btn btn-icon"
                        onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" />
                            <line x1="21" y1="21" x2="16.65" y2="16.65" />
                            <line x1="8" y1="11" x2="14" y2="11" />
                        </svg>
                    </button>
                    <span className="zoom-level">{Math.round(zoom * 100)}%</span>
                    <button
                        className="btn btn-icon"
                        onClick={() => setZoom(z => Math.min(3, z + 0.25))}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" />
                            <line x1="21" y1="21" x2="16.65" y2="16.65" />
                            <line x1="11" y1="8" x2="11" y2="14" />
                            <line x1="8" y1="11" x2="14" y2="11" />
                        </svg>
                    </button>

                    <div className="toolbar-divider" />

                    <label className="toggle-labels">
                        <input
                            type="checkbox"
                            checked={showLabels}
                            onChange={(e) => setShowLabels(e.target.checked)}
                        />
                        <span>Labels</span>
                    </label>
                </div>

                <div className="toolbar-right">
                    <select
                        className="class-filter"
                        value={selectedClass}
                        onChange={(e) => setSelectedClass(e.target.value)}
                    >
                        <option value="all">All Classes</option>
                        {classes.map(cls => (
                            <option key={cls} value={cls}>{cls}</option>
                        ))}
                    </select>

                    <button className="btn btn-primary" onClick={onExport}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7,10 12,15 17,10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        Export
                    </button>
                </div>
            </div>

            <div className="viewer-content">
                {/* Image Gallery */}
                <div className="image-gallery glass-card">
                    <div className="gallery-header">
                        <h4>Images ({images.length})</h4>
                    </div>
                    <div className="gallery-list">
                        {images.map((img, index) => {
                            const imgName = img.split(/[/\\]/).pop()
                            const ann = allAnnotations[img] || {}
                            const detections = ann.boxes?.length || 0

                            return (
                                <button
                                    key={index}
                                    className={`gallery-item ${selectedImage === index ? 'active' : ''}`}
                                    onClick={() => setSelectedImage(index)}
                                >
                                    <div className="gallery-thumb">
                                        <img
                                            src={`${API_URL}/uploads/${session?.session_id}/${imgName}`}
                                            alt={imgName}
                                        />
                                    </div>
                                    <div className="gallery-info">
                                        <span className="gallery-name">{imgName}</span>
                                        <span className="gallery-count">{detections} detections</span>
                                    </div>
                                </button>
                            )
                        })}
                    </div>
                </div>

                {/* Main Canvas */}
                <div className="canvas-container">
                    <div
                        className="canvas-wrapper"
                        style={{ transform: `scale(${zoom})` }}
                    >
                        <canvas ref={canvasRef} />
                    </div>
                </div>

                {/* Annotations Panel */}
                <div className="annotations-panel glass-card">
                    <div className="panel-header">
                        <h4>Detections</h4>
                        <span className="detection-count">{currentAnnotation.boxes?.length || 0}</span>
                    </div>

                    <div className="detection-list">
                        {currentAnnotation.boxes?.map((box, index) => {
                            const label = currentAnnotation.labels?.[index] || 'unknown'
                            const score = currentAnnotation.scores?.[index] || 0
                            const color = getClassColor(label)

                            return (
                                <div key={index} className="detection-item">
                                    <div
                                        className="detection-color"
                                        style={{ background: color }}
                                    />
                                    <div className="detection-info">
                                        <span className="detection-label">{label}</span>
                                        <span className="detection-score">{(score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            )
                        })}

                        {(!currentAnnotation.boxes || currentAnnotation.boxes.length === 0) && (
                            <div className="no-detections">
                                No objects detected in this image
                            </div>
                        )}
                    </div>

                    <div className="panel-stats">
                        <div className="panel-stat">
                            <span className="stat-value">{totalDetections}</span>
                            <span className="stat-label">Total Detections</span>
                        </div>
                        <div className="panel-stat">
                            <span className="stat-value">{(avgConfidence * 100).toFixed(1)}%</span>
                            <span className="stat-label">Avg Confidence</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ViewerPage
