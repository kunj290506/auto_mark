import React, { useState, useEffect, useRef, useCallback } from 'react'
import { API_URL } from '../config'
import './ViewerPage.css'

const IMAGES_PER_PAGE = 50

// UI 3: Confidence score color coding
const getConfidenceColor = (score) => {
    if (score >= 0.7) return '#059669'    // green
    if (score >= 0.4) return '#ea580c'    // orange
    return '#dc2626'                       // red
}

const getConfidenceBg = (score) => {
    if (score >= 0.7) return 'rgba(5,150,105,0.1)'
    if (score >= 0.4) return 'rgba(234,88,12,0.1)'
    return 'rgba(220,38,38,0.1)'
}

// Class color map
const CLASS_COLORS = [
    '#2563eb', '#7c3aed', '#db2777', '#059669', '#ea580c',
    '#dc2626', '#0891b2', '#6366f1', '#14b8a6', '#f59e0b',
    '#ec4899', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'
]

function ViewerPage({ session, annotations, onExport, onBack }) {
    const canvasRef = useRef(null)
    const [currentIndex, setCurrentIndex] = useState(0)
    // Bug 17: Zoom state + canvas transform
    const [zoom, setZoom] = useState(1)
    const [showLabels, setShowLabels] = useState(true)
    const [selectedAnnotation, setSelectedAnnotation] = useState(null)
    // Bug 18: Pagination
    const [currentPage, setCurrentPage] = useState(0)
    // UI 10: Class panel
    const [classFilter, setClassFilter] = useState(null)
    const [hiddenClasses, setHiddenClasses] = useState(new Set())

    const images = annotations?.images || []
    const totalPages = Math.ceil(images.length / IMAGES_PER_PAGE)
    const pageImages = images.slice(currentPage * IMAGES_PER_PAGE, (currentPage + 1) * IMAGES_PER_PAGE)

    const currentImage = images[currentIndex]
    const imageAnnotations = currentImage ? annotations?.annotations?.[currentImage] : null

    // Filter annotations by class
    const filteredAnnotations = imageAnnotations ? {
        ...imageAnnotations,
        boxes: (imageAnnotations.boxes || []).filter((_, i) => {
            const label = imageAnnotations.labels?.[i]
            if (hiddenClasses.has(label)) return false
            if (classFilter && label !== classFilter) return false
            return true
        }),
        labels: (imageAnnotations.labels || []).filter((label) => {
            if (hiddenClasses.has(label)) return false
            if (classFilter && label !== classFilter) return false
            return true
        }),
        scores: (imageAnnotations.scores || []).filter((_, i) => {
            const label = imageAnnotations.labels?.[i]
            if (hiddenClasses.has(label)) return false
            if (classFilter && label !== classFilter) return false
            return true
        })
    } : null

    // UI 10: Compute class stats
    const classStats = {}
    Object.values(annotations?.annotations || {}).forEach(ann => {
        (ann.labels || []).forEach(label => {
            classStats[label] = (classStats[label] || 0) + 1
        })
    })
    const sortedClasses = Object.entries(classStats).sort((a, b) => b[1] - a[1])

    // UI 2: Keyboard shortcuts
    useEffect(() => {
        const handleKey = (e) => {
            if (e.target.tagName === 'INPUT') return
            switch (e.key) {
                case 'ArrowRight':
                    e.preventDefault()
                    setCurrentIndex(i => Math.min(i + 1, images.length - 1))
                    break
                case 'ArrowLeft':
                    e.preventDefault()
                    setCurrentIndex(i => Math.max(i - 1, 0))
                    break
                case '+':
                case '=':
                    e.preventDefault()
                    setZoom(z => Math.min(z + 0.25, 5))
                    break
                case '-':
                    e.preventDefault()
                    setZoom(z => Math.max(z - 0.25, 0.25))
                    break
                case 'h':
                    setShowLabels(v => !v)
                    break
                case 'e':
                    onExport()
                    break
                case '0':
                    setZoom(1)
                    break
            }
        }
        window.addEventListener('keydown', handleKey)
        return () => window.removeEventListener('keydown', handleKey)
    }, [images.length, onExport])

    // Draw annotations on canvas
    useEffect(() => {
        if (!currentImage || !canvasRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const img = new Image()
        img.crossOrigin = 'anonymous'

        img.onload = () => {
            canvas.width = img.width
            canvas.height = img.height
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.drawImage(img, 0, 0)

            if (!filteredAnnotations) return

            const boxes = filteredAnnotations.boxes || []
            const labels = filteredAnnotations.labels || []
            const scores = filteredAnnotations.scores || []

            boxes.forEach((box, i) => {
                const label = labels[i] || 'object'
                const score = scores[i] || 0
                const classIdx = sortedClasses.findIndex(([c]) => c === label)
                const color = CLASS_COLORS[classIdx % CLASS_COLORS.length] || '#2563eb'

                ctx.strokeStyle = color
                ctx.lineWidth = 3

                const x = box.x1 || box.x * img.width
                const y = box.y1 || box.y * img.height
                const w = (box.x2 || (box.x + box.width) * img.width) - x
                const h = (box.y2 || (box.y + box.height) * img.height) - y

                // UI 1: Shape-aware annotations
                if (box.shape_type === 'circle') {
                    const cx = x + w / 2
                    const cy = y + h / 2
                    const rx = w / 2
                    const ry = h / 2
                    ctx.beginPath()
                    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
                    ctx.stroke()

                    ctx.fillStyle = color + '18'
                    ctx.fill()
                } else {
                    ctx.strokeRect(x, y, w, h)
                    ctx.fillStyle = color + '18'
                    ctx.fillRect(x, y, w, h)
                }

                // Draw label + score
                if (showLabels) {
                    const text = `${label} ${(score * 100).toFixed(0)}%`
                    ctx.font = 'bold 14px Inter, sans-serif'
                    const metrics = ctx.measureText(text)
                    const pad = 4

                    ctx.fillStyle = color
                    ctx.fillRect(x, y - 22, metrics.width + pad * 2, 22)

                    ctx.fillStyle = '#ffffff'
                    ctx.fillText(text, x + pad, y - 6)
                }

                // Highlight selected
                if (selectedAnnotation === i) {
                    ctx.strokeStyle = '#fbbf24'
                    ctx.lineWidth = 4
                    ctx.setLineDash([6, 4])
                    if (box.shape_type === 'circle') {
                        const cx = x + w / 2
                        const cy = y + h / 2
                        ctx.beginPath()
                        ctx.ellipse(cx, cy, w / 2 + 2, h / 2 + 2, 0, 0, Math.PI * 2)
                        ctx.stroke()
                    } else {
                        ctx.strokeRect(x - 2, y - 2, w + 4, h + 4)
                    }
                    ctx.setLineDash([])
                }
            })
        }

        // Bug 19: CORS error handler
        img.onerror = () => {
            canvas.width = 600
            canvas.height = 400
            const ctx2 = canvas.getContext('2d')
            ctx2.fillStyle = '#f8fafc'
            ctx2.fillRect(0, 0, 600, 400)
            ctx2.fillStyle = '#94a3b8'
            ctx2.font = '16px Inter, sans-serif'
            ctx2.textAlign = 'center'
            ctx2.fillText('Image could not be loaded (CORS or file missing)', 300, 200)
        }

        const filename = currentImage.split(/[/\\]/).pop()
        img.src = `${API_URL}/uploads/${session.session_id}/${filename}`

    }, [currentIndex, currentImage, filteredAnnotations, showLabels, selectedAnnotation, zoom, classFilter, hiddenClasses])

    // Bug 17: Mouse wheel zoom
    const handleWheel = useCallback((e) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            setZoom(z => {
                const delta = e.deltaY > 0 ? -0.1 : 0.1
                return Math.max(0.25, Math.min(5, z + delta))
            })
        }
    }, [])

    const getDetectionCount = (imgPath) => {
        const ann = annotations?.annotations?.[imgPath]
        return ann?.boxes?.length || 0
    }

    const hasLowConfidence = (imgPath) => {
        const ann = annotations?.annotations?.[imgPath]
        return (ann?.scores || []).some(s => s < 0.4)
    }

    return (
        <div className="viewer-page">
            <div className="viewer-layout">
                {/* Sidebar - Thumbnails */}
                <div className="viewer-sidebar">
                    <div className="sidebar-header">
                        <h4>Images ({images.length})</h4>
                        <span className="badge badge-primary">{currentPage + 1}/{totalPages || 1}</span>
                    </div>

                    <div className="thumbnail-list">
                        {pageImages.map((img, i) => {
                            const globalIdx = currentPage * IMAGES_PER_PAGE + i
                            const filename = img.split(/[/\\]/).pop()
                            const count = getDetectionCount(img)
                            return (
                                <div
                                    key={globalIdx}
                                    className={`thumbnail-item ${globalIdx === currentIndex ? 'active' : ''}`}
                                    onClick={() => setCurrentIndex(globalIdx)}
                                >
                                    <img
                                        src={`${API_URL}/uploads/${session.session_id}/${filename}`}
                                        alt={filename}
                                        loading="lazy"
                                    />
                                    <span className="thumb-filename">{filename}</span>
                                    {/* UI 4: Detection count badge */}
                                    <span className="detection-badge">{count}</span>
                                    {hasLowConfidence(img) && (
                                        <span className="warning-badge" title="Low confidence detections">!</span>
                                    )}
                                </div>
                            )
                        })}
                    </div>

                    {/* Bug 18: Pagination controls */}
                    {totalPages > 1 && (
                        <div className="pagination-controls">
                            <button
                                className="btn btn-sm btn-ghost"
                                disabled={currentPage === 0}
                                onClick={() => setCurrentPage(p => p - 1)}
                            >Prev</button>
                            <span className="page-info">{currentPage + 1} / {totalPages}</span>
                            <button
                                className="btn btn-sm btn-ghost"
                                disabled={currentPage >= totalPages - 1}
                                onClick={() => setCurrentPage(p => p + 1)}
                            >Next</button>
                        </div>
                    )}
                </div>

                {/* Canvas Area */}
                <div className="viewer-canvas-area" onWheel={handleWheel}>
                    <div className="canvas-toolbar">
                        <div className="toolbar-group">
                            <button className="btn btn-sm btn-ghost" onClick={() => setZoom(z => Math.max(z - 0.25, 0.25))}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                                    <circle cx="11" cy="11" r="8" />
                                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                                    <line x1="8" y1="11" x2="14" y2="11" />
                                </svg>
                            </button>
                            <span className="zoom-display">{Math.round(zoom * 100)}%</span>
                            <button className="btn btn-sm btn-ghost" onClick={() => setZoom(z => Math.min(z + 0.25, 5))}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                                    <circle cx="11" cy="11" r="8" />
                                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                                    <line x1="11" y1="8" x2="11" y2="14" />
                                    <line x1="8" y1="11" x2="14" y2="11" />
                                </svg>
                            </button>
                            <button className="btn btn-sm btn-ghost" onClick={() => setZoom(1)}>Fit</button>
                        </div>

                        <div className="toolbar-group">
                            <button className={`btn btn-sm ${showLabels ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setShowLabels(v => !v)}>
                                Labels
                            </button>
                            <button className="btn btn-sm btn-ghost" onClick={() => setCurrentIndex(i => Math.max(i - 1, 0))}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                                    <polyline points="15,18 9,12 15,6" />
                                </svg>
                            </button>
                            <span className="image-counter">{currentIndex + 1} / {images.length}</span>
                            <button className="btn btn-sm btn-ghost" onClick={() => setCurrentIndex(i => Math.min(i + 1, images.length - 1))}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                                    <polyline points="9,18 15,12 9,6" />
                                </svg>
                            </button>
                        </div>
                    </div>

                    {/* Bug 17: Scrollable container with zoom transform */}
                    <div className="canvas-scroll-container">
                        <canvas
                            ref={canvasRef}
                            className="annotation-canvas"
                            style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
                        />
                    </div>
                </div>

                {/* Right Panel - Annotations + Classes */}
                <div className="viewer-info-panel">
                    {/* UI 10: Class Management */}
                    <div className="class-panel">
                        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h4>Classes</h4>
                            {classFilter && (
                                <button className="btn btn-sm btn-ghost" onClick={() => setClassFilter(null)}>Clear</button>
                            )}
                        </div>
                        <div className="class-list">
                            {sortedClasses.map(([cls, count], i) => (
                                <div
                                    key={cls}
                                    className={`class-item ${classFilter === cls ? 'active' : ''} ${hiddenClasses.has(cls) ? 'hidden-class' : ''}`}
                                >
                                    <span
                                        className="class-dot"
                                        style={{ background: CLASS_COLORS[i % CLASS_COLORS.length] }}
                                    />
                                    <span className="class-name" onClick={() => setClassFilter(classFilter === cls ? null : cls)}>
                                        {cls}
                                    </span>
                                    <span className="class-count badge badge-primary">{count}</span>
                                    <button
                                        className="btn btn-sm btn-icon class-hide-btn"
                                        title={hiddenClasses.has(cls) ? 'Show' : 'Hide'}
                                        onClick={() => {
                                            const next = new Set(hiddenClasses)
                                            if (next.has(cls)) next.delete(cls)
                                            else next.add(cls)
                                            setHiddenClasses(next)
                                        }}
                                    >
                                        {hiddenClasses.has(cls) ? (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                                                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                                                <line x1="1" y1="1" x2="23" y2="23" />
                                            </svg>
                                        ) : (
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                                                <circle cx="12" cy="12" r="3" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Annotations list for current image */}
                    <div className="annotations-panel">
                        <h4>Detections ({filteredAnnotations?.boxes?.length || 0})</h4>
                        <div className="annotation-list">
                            {(filteredAnnotations?.boxes || []).map((box, i) => {
                                const label = filteredAnnotations.labels?.[i] || 'object'
                                const score = filteredAnnotations.scores?.[i] || 0
                                const classIdx = sortedClasses.findIndex(([c]) => c === label)
                                return (
                                    <div
                                        key={i}
                                        className={`annotation-item ${selectedAnnotation === i ? 'selected' : ''}`}
                                        onClick={() => setSelectedAnnotation(selectedAnnotation === i ? null : i)}
                                    >
                                        <span
                                            className="ann-color-dot"
                                            style={{ background: CLASS_COLORS[classIdx % CLASS_COLORS.length] }}
                                        />
                                        <span className="ann-label">{label}</span>
                                        {/* UI 1: Shape type indicator */}
                                        {box.shape_type === 'circle' && (
                                            <span className="shape-badge" title="Circular detection">C</span>
                                        )}
                                        {/* UI 3: Color coded confidence */}
                                        <span
                                            className="ann-score"
                                            style={{ color: getConfidenceColor(score), background: getConfidenceBg(score) }}
                                        >
                                            {(score * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                )
                            })}
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="panel-actions">
                        <button className="btn btn-primary btn-lg" style={{ width: '100%' }} onClick={onExport}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7,10 12,15 17,10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            Export Annotations
                        </button>
                        <button className="btn btn-secondary" style={{ width: '100%' }} onClick={onBack}>
                            Back to Configure
                        </button>
                    </div>

                    {/* UI 2: Keyboard shortcuts hint */}
                    <div className="shortcuts-hint">
                        <span>Arrow keys: Navigate</span>
                        <span>+/-: Zoom</span>
                        <span>H: Labels</span>
                        <span>E: Export</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ViewerPage
