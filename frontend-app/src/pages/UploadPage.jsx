import React, { useState, useRef, useCallback } from 'react'
import './UploadPage.css'

const API_URL = 'http://localhost:8000'

function UploadPage({ onComplete }) {
    const [isDragOver, setIsDragOver] = useState(false)
    const [file, setFile] = useState(null)
    const [uploading, setUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [error, setError] = useState(null)
    const [previewImages, setPreviewImages] = useState([])
    const fileInputRef = useRef(null)

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(false)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setIsDragOver(false)

        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && droppedFile.name.endsWith('.zip')) {
            setFile(droppedFile)
            setError(null)
        } else {
            setError('Please upload a ZIP file')
        }
    }, [])

    const handleFileSelect = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            setFile(selectedFile)
            setError(null)
        }
    }

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes'
        const k = 1024
        const sizes = ['Bytes', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(k))
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }

    const handleUpload = async () => {
        if (!file) return

        setUploading(true)
        setUploadProgress(0)
        setError(null)

        try {
            const formData = new FormData()
            formData.append('file', file)

            // Use XMLHttpRequest for real upload progress
            const result = await new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest()

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const progress = Math.round((e.loaded / e.total) * 100)
                        setUploadProgress(progress)
                    }
                })

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            resolve(JSON.parse(xhr.responseText))
                        } catch (e) {
                            reject(new Error('Invalid server response'))
                        }
                    } else {
                        try {
                            const errorData = JSON.parse(xhr.responseText)
                            reject(new Error(errorData.detail || 'Upload failed'))
                        } catch (e) {
                            reject(new Error(`Upload failed: ${xhr.status}`))
                        }
                    }
                })

                xhr.addEventListener('error', () => reject(new Error('Network error')))
                xhr.addEventListener('timeout', () => reject(new Error('Upload timeout')))

                xhr.open('POST', `${API_URL}/api/upload`)
                xhr.timeout = 600000 // 10 minute timeout for large files
                xhr.send(formData)
            })

            // Generate preview URLs for images
            // Handle both Windows (\\) and Unix (/) path separators
            const previews = result.images.slice(0, 8).map((img, i) => {
                const filename = img.split(/[/\\]/).pop()
                return {
                    id: i,
                    path: img,
                    url: `${API_URL}/uploads/${result.session_id}/${filename}`
                }
            })

            setPreviewImages(previews)

            // Small delay to show 100% progress
            setTimeout(() => {
                onComplete(result)
            }, 500)

        } catch (err) {
            setError(err.message)
            setUploading(false)
            setUploadProgress(0)
        }
    }

    const handleRemoveFile = () => {
        setFile(null)
        setError(null)
        setPreviewImages([])
        if (fileInputRef.current) {
            fileInputRef.current.value = ''
        }
    }

    return (
        <div className="upload-page">
            <div className="upload-container animate-slideUp">
                {!file ? (
                    <div
                        className={`dropzone glass-card ${isDragOver ? 'drag-over' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".zip"
                            onChange={handleFileSelect}
                            hidden
                        />

                        <div className="dropzone-icon">
                            <svg viewBox="0 0 64 64" fill="none">
                                <rect x="8" y="8" width="48" height="48" rx="12" stroke="currentColor" strokeWidth="2" strokeDasharray="4 2" />
                                <path d="M32 20v24" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                <path d="M24 28l8-8 8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                <path d="M20 44h24" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        </div>

                        <h3 className="dropzone-title">
                            Drag & drop your ZIP file here
                        </h3>
                        <p className="dropzone-subtitle">
                            or click to browse your files
                        </p>

                        <div className="dropzone-info">
                            <span className="info-badge">
                                <svg viewBox="0 0 16 16" fill="currentColor">
                                    <path d="M4 1.5a.5.5 0 0 0-.5.5v12a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5V4.707L9.293 1.5H4z" />
                                    <path d="M9 1.5v3a.5.5 0 0 0 .5.5h3" />
                                </svg>
                                ZIP files only
                            </span>
                            <span className="info-badge">
                                <svg viewBox="0 0 16 16" fill="currentColor">
                                    <path d="M4 11a4 4 0 0 1 8 0v1H4v-1z" />
                                    <path d="M8 1a4 4 0 0 1 4 4v6H4V5a4 4 0 0 1 4-4z" />
                                </svg>
                                Max 1GB
                            </span>
                            <span className="info-badge">
                                <svg viewBox="0 0 16 16" fill="currentColor">
                                    <path d="M.002 3a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-12a2 2 0 0 1-2-2V3zm1.5 0a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h12a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5h-12z" />
                                </svg>
                                JPG, PNG, BMP
                            </span>
                        </div>
                    </div>
                ) : (
                    <div className="file-preview glass-card">
                        <div className="preview-header">
                            <div className="file-icon">
                                <svg viewBox="0 0 48 48" fill="none">
                                    <rect x="8" y="4" width="32" height="40" rx="4" fill="url(#zipGradient)" />
                                    <path d="M18 12h12M18 18h12M18 24h12M18 30h8" stroke="white" strokeWidth="2" strokeLinecap="round" />
                                    <defs>
                                        <linearGradient id="zipGradient" x1="8" y1="4" x2="40" y2="44">
                                            <stop stopColor="#6366f1" />
                                            <stop offset="1" stopColor="#8b5cf6" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                            </div>

                            <div className="file-info">
                                <h4 className="file-name">{file.name}</h4>
                                <p className="file-size">{formatFileSize(file.size)}</p>
                            </div>

                            {!uploading && (
                                <button className="remove-btn" onClick={handleRemoveFile}>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            )}
                        </div>

                        {uploading && (
                            <div className="upload-progress">
                                <div className="progress-bar">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${uploadProgress}%` }}
                                    />
                                </div>
                                <span className="progress-text">
                                    {uploadProgress === 0
                                        ? 'Preparing upload... (large files may take a moment)'
                                        : `${uploadProgress}% uploaded`}
                                </span>
                            </div>
                        )}

                        {previewImages.length > 0 && (
                            <div className="image-previews">
                                <h5 className="previews-title">Image Preview</h5>
                                <div className="previews-grid">
                                    {previewImages.map((img) => (
                                        <div key={img.id} className="preview-item">
                                            <img src={img.url} alt={`Preview ${img.id + 1}`} />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {!uploading && (
                            <button className="btn btn-primary btn-lg upload-btn" onClick={handleUpload}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="17,8 12,3 7,8" />
                                    <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                                Upload & Continue
                            </button>
                        )}
                    </div>
                )}

                {error && (
                    <div className="error-message animate-slideUp">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        {error}
                    </div>
                )}
            </div>

            <div className="upload-features">
                <div className="feature-card glass-card animate-slideUp delay-100">
                    <div className="feature-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <polyline points="12,6 12,12 16,14" />
                        </svg>
                    </div>
                    <h4>Fast Processing</h4>
                    <p>2-5 seconds per image with GPU acceleration</p>
                </div>

                <div className="feature-card glass-card animate-slideUp delay-200">
                    <div className="feature-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                            <polyline points="3.27,6.96 12,12.01 20.73,6.96" />
                            <line x1="12" y1="22.08" x2="12" y2="12" />
                        </svg>
                    </div>
                    <h4>Multiple Formats</h4>
                    <p>Export to COCO, YOLO, VOC, and Roboflow</p>
                </div>

                <div className="feature-card glass-card animate-slideUp delay-300">
                    <div className="feature-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <h4>Zero-Shot Detection</h4>
                    <p>No training needed, describe what you want to find</p>
                </div>
            </div>
        </div>
    )
}

export default UploadPage
