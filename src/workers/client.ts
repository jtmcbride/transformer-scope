import { EngineError } from '../engine/contracts';
import type { Command, Commands, WorkerRequest, WorkerResponse } from './protocol';

export class ModelClient {
  private readonly worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' });
  private nextId = 0;
  private disposed = false;
  private pending = new Map<number, { resolve: (result: unknown) => void; reject: (error: Error) => void }>();

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data, pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      if (response.ok) pending.resolve(response.result);
      else pending.reject(new EngineError(response.error.code, response.error.message));
    };
    this.worker.onerror = event => {
      event.preventDefault();
      this.fail(new EngineError('STALE_SESSION', `Model worker stopped: ${event.message}`));
    };
    this.worker.onmessageerror = () => this.fail(new EngineError('STALE_SESSION', 'Model worker could not read a response.'));
  }

  request<K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> {
    if (this.disposed) return Promise.reject(new EngineError('STALE_SESSION', 'This worker session has ended.'));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: result => resolve(result as Commands[K]['response']), reject });
      try { this.worker.postMessage({ id, command, payload } as WorkerRequest); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }

  private fail(error: EngineError): void {
    this.disposed = true; this.worker.terminate();
    this.pending.forEach(pending => pending.reject(error)); this.pending.clear();
  }
  dispose(): void { this.fail(new EngineError('STALE_SESSION', 'This worker session has ended.')); }
}
