import { createRoot } from 'react-dom/client';
import App from './App';
import './style.css';
import './flow.css';
import './trace.css';
import './visual.css';
import './continuation.css';

createRoot(document.getElementById('root')!).render(<App />);
