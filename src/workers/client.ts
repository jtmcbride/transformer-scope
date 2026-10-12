import { EngineError } from '../engine/contracts';
import type { Command, Commands, WorkerNotice, WorkerRequest, WorkerResponse } from './protocol';

export class ModelClient {
  private readonly worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' });
  private nextId = 0;
  private ended: EngineError | null = null;
  private pending = new Map<number, { resolve: (result: unknown) => void; reject: (error: Error) => void }>();
  /** Called once if the session ends without dispose(): a worker crash or a lost GPU context. */
  onEnded: ((error: EngineError) => void) | null = null;

  constructor() {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse | WorkerNotice>) => {
      const message = event.data;
      if ('notice' in message) { this.fail(new EngineError('STALE_SESSION', message.message)); return; }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.ok) pending.resolve(message.result);
      else pending.reject(new EngineError(message.error.code, message.error.message));
    };
    this.worker.onerror = event => {
      event.preventDefault();
      this.fail(new EngineError('STALE_SESSION', `Model worker stopped: ${event.message}`));
    };
    this.worker.onmessageerror = () => this.fail(new EngineError('STALE_SESSION', 'Model worker could not read a response.'));
  }

  /** The reason this session ended, or null while it can still compute. */
  get endReason(): EngineError | null { return this.ended; }

  request<K extends Command>(command: K, payload: Commands[K]['request']): Promise<Commands[K]['response']> {
    if (this.ended) return Promise.reject(this.ended);
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: result => resolve(result as Commands[K]['response']), reject });
      try { this.worker.postMessage({ id, command, payload } as WorkerRequest); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }

  private fail(error: EngineError, notify = true): void {
    if (this.ended) return;
    this.ended = error; this.worker.terminate();
    this.pending.forEach(pending => pending.reject(error)); this.pending.clear();
    if (notify) this.onEnded?.(error);
  }
  dispose(): void { this.fail(new EngineError('STALE_SESSION', 'This worker session has ended.'), false); }
}
