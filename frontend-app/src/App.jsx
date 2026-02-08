import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import UploadPage from './pages/UploadPage'
import ConfigurePage from './pages/ConfigurePage'
import ProcessingPage from './pages/ProcessingPage'
import ViewerPage from './pages/ViewerPage'
import ExportPage from './pages/ExportPage'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState('upload')
  const [session, setSession] = useState(null)
  const [config, setConfig] = useState({
    objects: [],
    boxThreshold: 0.35,
    textThreshold: 0.25,
    useSam: false,
    exportFormat: 'coco'
  })
  const [annotations, setAnnotations] = useState(null)

  // Navigation handler
  const navigate = (page) => {
    setCurrentPage(page)
  }

  // Handle upload completion
  const handleUploadComplete = (sessionData) => {
    setSession(sessionData)
    navigate('configure')
  }

  // Handle configuration submit
  const handleConfigSubmit = (newConfig) => {
    setConfig(newConfig)
    navigate('processing')
  }

  // Handle processing completion
  const handleProcessingComplete = (annotationData) => {
    setAnnotations(annotationData)
    navigate('viewer')
  }

  // Render current page
  const renderPage = () => {
    switch (currentPage) {
      case 'upload':
        return <UploadPage onComplete={handleUploadComplete} />
      case 'configure':
        return (
          <ConfigurePage
            session={session}
            config={config}
            onSubmit={handleConfigSubmit}
            onBack={() => navigate('upload')}
          />
        )
      case 'processing':
        return (
          <ProcessingPage
            session={session}
            config={config}
            onComplete={handleProcessingComplete}
            onCancel={() => navigate('configure')}
          />
        )
      case 'viewer':
        return (
          <ViewerPage
            session={session}
            annotations={annotations}
            onExport={() => navigate('export')}
            onBack={() => navigate('configure')}
          />
        )
      case 'export':
        return (
          <ExportPage
            session={session}
            annotations={annotations}
            config={config}
            onBack={() => navigate('viewer')}
            onNewProject={() => {
              setSession(null)
              setAnnotations(null)
              navigate('upload')
            }}
          />
        )
      default:
        return <UploadPage onComplete={handleUploadComplete} />
    }
  }

  return (
    <div className="app">
      <Sidebar currentPage={currentPage} onNavigate={navigate} session={session} />
      <main className="main-content">
        <Header session={session} currentPage={currentPage} />
        <div className="page-content">
          {renderPage()}
        </div>
      </main>
    </div>
  )
}

export default App
