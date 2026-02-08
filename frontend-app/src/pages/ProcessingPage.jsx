import { useState, useEffect, useRef } from 'react'
import './ProcessingPage.css'

// Use relative URLs - Vite proxy will forward to backend
const API_URL = ''

function ProcessingPage({ session, config, onComplete, onCancel }) {
    const [status, setStatus] = useState('connecting')
    const [progress, setProgress] = useState(0)
    const [currentImage, setCurrentImage] = useState(null)
    const [processedCount, setProcessedCount] = useState(0)
    const [detectionCount, setDetectionCount] = useState(0)
    const [startTime, setStartTime] = useState(null)
    const [eta, setEta] = useState(null)
    const pollingRef = useRef(null)
    const hasStarted = useRef(false)

    useEffect(() => {
        if (!session || hasStarted.current) return
        hasStarted.current = true

        startProcessing()

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current)
            }
        }
    }, [session])

    const startProcessing = async () => {
        setStartTime(Date.now())
        setStatus('processing')

        // Start annotation job first
        await startAnnotation()

        // Then start polling for progress
        startPolling()
    }

    const startAnnotation = async () => {
        try {
            const response = await fetch(`${API_URL}/api/annotate/${session.session_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    objects: config.objects,
                    box_threshold: config.boxThreshold,
                    text_threshold: config.textThreshold,
                    use_sam: config.useSam || false,
                    export_format: config.exportFormat
                })
            })

            if (!response.ok) {
                const errorData = await response.json()
                throw new Error(errorData.detail || 'Failed to start annotation')
            }

            return true
        } catch (error) {
            console.error('Error starting annotation:', error)
            setStatus('error')
            return false
        }
    }

    const startPolling = () => {
        pollingRef.current = setInterval(async () => {
            try {
                const response = await fetch(`${API_URL}/api/status/${session.session_id}`)
                const data = await response.json()

                const percentage = data.progress || 0
                setProgress(percentage)
                setProcessedCount(data.processed_count || 0)

                // Calculate ETA
                if (startTime && data.processed_count > 0) {
                    const elapsed = Date.now() - startTime
                    const remaining = (elapsed / data.processed_count) * (session.image_count - data.processed_count)
                    setEta(Math.ceil(remaining / 1000))
                }

                if (data.status === 'completed') {
                    clearInterval(pollingRef.current)
                    setStatus('completed')
                    setProgress(100)
                    setTimeout(() => handleComplete(), 1000)
                } else if (data.status === 'error') {
                    clearInterval(pollingRef.current)
                    setStatus('error')
                }
            } catch (error) {
                console.error('Polling error:', error)
            }
        }, 500) // Poll every 500ms for smoother updates
    }

    const handleComplete = async () => {
        try {
            const response = await fetch(`${API_URL}/api/annotations/${session.session_id}`)
            const data = await response.json()

            // Count total detections
            const totalDetections = Object.values(data.annotations || {}).reduce(
                (sum, ann) => sum + (ann.boxes?.length || 0), 0
            )
            setDetectionCount(totalDetections)

            onComplete(data)
        } catch (error) {
            console.error('Error fetching annotations:', error)
            setStatus('error')
        }
    }

    const formatEta = (seconds) => {
        if (!seconds) return '--:--'
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    const formatElapsed = () => {
        if (!startTime) return '0:00'
        const elapsed = Math.floor((Date.now() - startTime) / 1000)
        const mins = Math.floor(elapsed / 60)
        const secs = elapsed % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    // Update elapsed time display
    const [, forceUpdate] = useState(0)
    useEffect(() => {
        if (status !== 'processing') return
        const interval = setInterval(() => {
            forceUpdate(n => n + 1) // Force re-render for elapsed time
        }, 1000)
        return () => clearInterval(interval)
    }, [status])

    const handleRetry = () => {
        hasStarted.current = false
        setStatus('connecting')
        setProgress(0)
        setProcessedCount(0)
        setDetectionCount(0)
        startProcessing()
    }

    return (
        <div className="processing-page">
            <div className="processing-container">
                {/* Main Progress Card */}
                <div className="progress-card glass-card animate-slideUp">
                    <div className="progress-visual">
                        <div className="progress-ring">
                            <svg viewBox="0 0 120 120">
                                <circle
                                    className="ring-bg"
                                    cx="60" cy="60" r="52"
                                    strokeWidth="8"
                                />
                                <circle
                                    className="ring-progress"
                                    cx="60" cy="60" r="52"
                                    strokeWidth="8"
                                    strokeDasharray={`${progress * 3.27} 327`}
                                    strokeLinecap="round"
                                />
                            </svg>
                            <div className="progress-center">
                                <span className="progress-value">{Math.round(progress)}%</span>
                                <span className="progress-label">
                                    {status === 'connecting' ? 'Starting...' :
                                        status === 'completed' ? 'Done!' :
                                            status === 'error' ? 'Error' : 'Processing'}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="progress-info">
                        <h2 className="progress-title">
                            {status === 'connecting' && 'Setting up...'}
                            {status === 'processing' && 'Annotating Images'}
                            {status === 'completed' && 'Annotation Complete!'}
                            {status === 'error' && 'An Error Occurred'}
                        </h2>
                        <p className="progress-subtitle">
                            {status === 'processing' && `Processing ${processedCount} of ${session?.image_count || 0} images`}
                            {status === 'completed' && `Successfully annotated ${session?.image_count || 0} images`}
                            {status === 'error' && 'Please try again or check your configuration'}
                        </p>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="stats-grid">
                    <div className="stat-card glass-card animate-slideUp delay-100">
                        <div className="stat-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" />
                                <circle cx="8.5" cy="8.5" r="1.5" />
                                <path d="M21 15l-5-5L5 21" />
                            </svg>
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{processedCount}</span>
                            <span className="stat-label">Images Processed</span>
                        </div>
                    </div>

                    <div className="stat-card glass-card animate-slideUp delay-200">
                        <div className="stat-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            </svg>
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{detectionCount}</span>
                            <span className="stat-label">Objects Detected</span>
                        </div>
                    </div>

                    <div className="stat-card glass-card animate-slideUp delay-300">
                        <div className="stat-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <polyline points="12,6 12,12 16,14" />
                            </svg>
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{formatElapsed()}</span>
                            <span className="stat-label">Elapsed Time</span>
                        </div>
                    </div>

                    <div className="stat-card glass-card animate-slideUp delay-400">
                        <div className="stat-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M12 2v4" />
                                <path d="M12 18v4" />
                                <path d="M4.93 4.93l2.83 2.83" />
                                <path d="M16.24 16.24l2.83 2.83" />
                                <path d="M2 12h4" />
                                <path d="M18 12h4" />
                                <path d="M4.93 19.07l2.83-2.83" />
                                <path d="M16.24 7.76l2.83-2.83" />
                            </svg>
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{formatEta(eta)}</span>
                            <span className="stat-label">Estimated Time</span>
                        </div>
                    </div>
                </div>

                {/* Current Image */}
                {currentImage && status === 'processing' && (
                    <div className="current-image glass-card animate-slideUp">
                        <span className="current-label">Currently Processing</span>
                        <span className="current-filename">{currentImage.split('/').pop()}</span>
                    </div>
                )}

                {/* Configuration Summary */}
                <div className="config-summary glass-card animate-slideUp delay-500">
                    <h4>Configuration</h4>
                    <div className="config-tags">
                        {config.objects.map((obj, i) => (
                            <span key={i} className="config-tag">🎯 {obj}</span>
                        ))}
                        <span className="config-tag">📊 {config.exportFormat.toUpperCase()}</span>
                        <span className="config-tag">🎚️ Box: {config.boxThreshold}</span>
                    </div>
                </div>

                {/* Actions */}
                {status === 'error' && (
                    <div className="processing-actions">
                        <button className="btn btn-secondary" onClick={onCancel}>
                            Back to Configure
                        </button>
                        <button className="btn btn-primary" onClick={handleRetry}>
                            Retry
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}

export default ProcessingPage
