import { createRoot } from 'react-dom/client';
import StudyApp from './StudyApp';
import '../validation/validation.css';
import './study.css';

createRoot(document.getElementById('root')!).render(<StudyApp />);
