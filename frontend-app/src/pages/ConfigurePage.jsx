import { useState } from 'react'
import './ConfigurePage.css'

function ConfigurePage({ session, config, onSubmit, onBack }) {
    const [localConfig, setLocalConfig] = useState({
        objects: config.objects.length > 0 ? config.objects : [''],
        boxThreshold: config.boxThreshold,
        textThreshold: config.textThreshold,
        useSam: config.useSam,
        exportFormat: config.exportFormat,
        minBoxSize: 10,
        removeOverlaps: true,
        skipDetection: false  // New: Skip object detection option
    })

    const handleObjectChange = (index, value) => {
        const newObjects = [...localConfig.objects]
        newObjects[index] = value
        setLocalConfig({ ...localConfig, objects: newObjects })
    }

    const addObject = () => {
        setLocalConfig({
            ...localConfig,
            objects: [...localConfig.objects, '']
        })
    }

    const removeObject = (index) => {
        const newObjects = localConfig.objects.filter((_, i) => i !== index)
        setLocalConfig({ ...localConfig, objects: newObjects.length ? newObjects : [''] })
    }

    const handleSkipToggle = (checked) => {
        setLocalConfig({
            ...localConfig,
            skipDetection: checked,
            // If skipping, use a generic "object" label
            objects: checked ? ['object'] : ['']
        })
    }

    const handleSubmit = () => {
        if (!localConfig.skipDetection) {
            const validObjects = localConfig.objects.filter(obj => obj.trim())
            if (validObjects.length === 0) {
                alert('Please add at least one object to detect, or enable "Skip Detection"')
                return
            }
            onSubmit({ ...localConfig, objects: validObjects })
        } else {
            // If skipping, use a generic detection prompt
            onSubmit({ ...localConfig, objects: ['object'] })
        }
    }

    const formatOptions = [
        { id: 'coco', name: 'COCO JSON', desc: 'Single JSON file for all annotations' },
        { id: 'yolo', name: 'YOLO', desc: 'Separate .txt files per image' },
        { id: 'voc', name: 'Pascal VOC', desc: 'XML format for each image' },
        { id: 'roboflow', name: 'Roboflow', desc: 'Compatible with Roboflow platform' }
    ]

    return (
        <div className="configure-page">
            <div className="config-grid">
                {/* Object Detection Section */}
                <section className="config-section glass-card animate-slideUp">
                    <div className="section-header">
                        <div className="section-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8" />
                                <path d="M21 21l-4.35-4.35" />
                            </svg>
                        </div>
                        <div>
                            <h3 className="section-title">Objects to Detect</h3>
                            <p className="section-desc">Describe what you want to find in your images</p>
                        </div>
                    </div>

                    {/* Skip Detection Toggle */}
                    <div className="skip-detection-box">
                        <label className="toggle-option skip-toggle">
                            <div className="toggle-info">
                                <span className="toggle-label">⏭️ Skip Object Description</span>
                                <span className="toggle-desc">Detect all objects automatically without specific labels</span>
                            </div>
                            <input
                                type="checkbox"
                                className="toggle-input"
                                checked={localConfig.skipDetection}
                                onChange={(e) => handleSkipToggle(e.target.checked)}
                            />
                            <span className="toggle-switch"></span>
                        </label>
                    </div>

                    {/* Object Inputs - Only show if not skipping */}
                    {!localConfig.skipDetection && (
                        <>
                            <div className="objects-list">
                                {localConfig.objects.map((obj, index) => (
                                    <div key={index} className="object-input-wrapper">
                                        <input
                                            type="text"
                                            className="input object-input"
                                            value={obj}
                                            onChange={(e) => handleObjectChange(index, e.target.value)}
                                            placeholder={`e.g., "football", "red car", "person wearing helmet"`}
                                        />
                                        {localConfig.objects.length > 1 && (
                                            <button
                                                className="remove-object-btn"
                                                onClick={() => removeObject(index)}
                                            >
                                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                    <line x1="18" y1="6" x2="6" y2="18" />
                                                    <line x1="6" y1="6" x2="18" y2="18" />
                                                </svg>
                                            </button>
                                        )}
                                    </div>
                                ))}

                                <button className="btn btn-secondary add-object-btn" onClick={addObject}>
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <line x1="12" y1="5" x2="12" y2="19" />
                                        <line x1="5" y1="12" x2="19" y2="12" />
                                    </svg>
                                    Add Another Object
                                </button>
                            </div>

                            <div className="tips-box">
                                <h4>💡 Tips for Better Detection</h4>
                                <ul>
                                    <li>Be specific: "red car" works better than "car"</li>
                                    <li>Add context: "person wearing helmet" finds specific people</li>
                                    <li>Use descriptive phrases: "round ball with black and white patches"</li>
                                </ul>
                            </div>
                        </>
                    )}

                    {/* Show message when skipping */}
                    {localConfig.skipDetection && (
                        <div className="skip-info-box">
                            <div className="skip-info-icon">🔍</div>
                            <div className="skip-info-content">
                                <h4>Auto-detect All Objects</h4>
                                <p>The model will attempt to detect all visible objects in your images without specific labels. Results will be labeled as "object".</p>
                            </div>
                        </div>
                    )}
                </section>

                {/* Thresholds Section */}
                <section className="config-section glass-card animate-slideUp delay-100">
                    <div className="section-header">
                        <div className="section-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <line x1="4" y1="21" x2="4" y2="14" />
                                <line x1="4" y1="10" x2="4" y2="3" />
                                <line x1="12" y1="21" x2="12" y2="12" />
                                <line x1="12" y1="8" x2="12" y2="3" />
                                <line x1="20" y1="21" x2="20" y2="16" />
                                <line x1="20" y1="12" x2="20" y2="3" />
                                <circle cx="4" cy="12" r="2" />
                                <circle cx="12" cy="10" r="2" />
                                <circle cx="20" cy="14" r="2" />
                            </svg>
                        </div>
                        <div>
                            <h3 className="section-title">Detection Thresholds</h3>
                            <p className="section-desc">Adjust sensitivity of the detection model</p>
                        </div>
                    </div>

                    <div className="threshold-controls">
                        <div className="threshold-control">
                            <div className="threshold-header">
                                <label>Box Confidence</label>
                                <span className="threshold-value">{localConfig.boxThreshold.toFixed(2)}</span>
                            </div>
                            <input
                                type="range"
                                className="slider"
                                min="0.1"
                                max="0.9"
                                step="0.05"
                                value={localConfig.boxThreshold}
                                onChange={(e) => setLocalConfig({ ...localConfig, boxThreshold: parseFloat(e.target.value) })}
                            />
                            <p className="threshold-desc">Higher = fewer but more accurate detections</p>
                        </div>

                        <div className="threshold-control">
                            <div className="threshold-header">
                                <label>Text Matching</label>
                                <span className="threshold-value">{localConfig.textThreshold.toFixed(2)}</span>
                            </div>
                            <input
                                type="range"
                                className="slider"
                                min="0.1"
                                max="0.9"
                                step="0.05"
                                value={localConfig.textThreshold}
                                onChange={(e) => setLocalConfig({ ...localConfig, textThreshold: parseFloat(e.target.value) })}
                            />
                            <p className="threshold-desc">Lower = more flexible text matching</p>
                        </div>
                    </div>

                    <div className="toggle-options">
                        <label className="toggle-option">
                            <div className="toggle-info">
                                <span className="toggle-label">Remove Overlapping Boxes</span>
                                <span className="toggle-desc">Apply NMS to reduce duplicates</span>
                            </div>
                            <input
                                type="checkbox"
                                className="toggle-input"
                                checked={localConfig.removeOverlaps}
                                onChange={(e) => setLocalConfig({ ...localConfig, removeOverlaps: e.target.checked })}
                            />
                            <span className="toggle-switch"></span>
                        </label>
                    </div>
                </section>

                {/* Export Format Section */}
                <section className="config-section glass-card animate-slideUp delay-200">
                    <div className="section-header">
                        <div className="section-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="7,10 12,15 17,10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                        </div>
                        <div>
                            <h3 className="section-title">Export Format</h3>
                            <p className="section-desc">Choose your preferred annotation format</p>
                        </div>
                    </div>

                    <div className="format-grid">
                        {formatOptions.map((format) => (
                            <label
                                key={format.id}
                                className={`format-option ${localConfig.exportFormat === format.id ? 'selected' : ''}`}
                            >
                                <input
                                    type="radio"
                                    name="format"
                                    value={format.id}
                                    checked={localConfig.exportFormat === format.id}
                                    onChange={(e) => setLocalConfig({ ...localConfig, exportFormat: e.target.value })}
                                />
                                <div className="format-content">
                                    <span className="format-name">{format.name}</span>
                                    <span className="format-desc">{format.desc}</span>
                                </div>
                                <div className="format-check">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polyline points="20,6 9,17 4,12" />
                                    </svg>
                                </div>
                            </label>
                        ))}
                    </div>
                </section>
            </div>

            {/* Session Info */}
            <div className="session-summary glass-card animate-slideUp delay-300">
                <div className="summary-stat">
                    <span className="stat-icon">📁</span>
                    <span className="stat-value">{session?.image_count || 0}</span>
                    <span className="stat-label">Images</span>
                </div>
                <div className="summary-stat">
                    <span className="stat-icon">🎯</span>
                    <span className="stat-value">
                        {localConfig.skipDetection ? 'Auto' : localConfig.objects.filter(o => o.trim()).length}
                    </span>
                    <span className="stat-label">Objects</span>
                </div>
                <div className="summary-stat">
                    <span className="stat-icon">📊</span>
                    <span className="stat-value">{localConfig.exportFormat.toUpperCase()}</span>
                    <span className="stat-label">Format</span>
                </div>
            </div>

            {/* Actions */}
            <div className="config-actions">
                <button className="btn btn-secondary" onClick={onBack}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="19" y1="12" x2="5" y2="12" />
                        <polyline points="12,19 5,12 12,5" />
                    </svg>
                    Back
                </button>
                <button className="btn btn-primary btn-lg" onClick={handleSubmit}>
                    Start Annotation
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5,3 19,12 5,21" fill="currentColor" />
                    </svg>
                </button>
            </div>
        </div>
    )
}

export default ConfigurePage
