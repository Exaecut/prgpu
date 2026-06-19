//! Background tasks with a **host-aware driver**, behind one surface.
//!
//! `BackgroundTask::poll` is the cooperative unit of work; how it's driven
//! depends on the host (set once at GlobalSetup via [`set_host`]):
//!
//! - **After Effects** — the AEGP idle hook
//!   (`RegisterNonAegpSuite::register_idle_hook`, the `background_task` SDK
//!   example) calls [`pump`] each idle tick on the **main thread**.
//! - **Premiere Pro** — `AEGP_RegisterIdleHook` lives in `AE_GeneralPlug.h`,
//!   which the Premiere SDK does **not** ship; Premiere never pumps it. So each
//!   task is driven on a dedicated **`std::thread`** instead, polling to
//!   completion off the main thread.
//!
//! Either way the task reports through the shared-state [`StatusSink`] and is
//! cancelled via the tokio-style [`CancelToken`]; the registry is the retrieval
//! system so [`cancel`]/[`cancel_tag`] can target tasks without an "active
//! task" global. (A `std::sync::mpsc` channel could replace the shared `Mutex`
//! if event streaming is ever needed; latest-status shared state is enough for
//! progress display.)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::Mutex;

static PUMP_SEEN: AtomicBool = AtomicBool::new(false);

// Host driver selection. 0 = unknown (default to worker thread — the safe
// choice, since only AE guarantees an idle pump), 1 = After Effects (idle
// pump), 2 = Premiere (worker thread).
const HOST_UNKNOWN: u8 = 0;
const HOST_AFTER_EFFECTS: u8 = 1;
const HOST_PREMIERE: u8 = 2;
static HOST: AtomicU8 = AtomicU8::new(HOST_UNKNOWN);

/// Called once at GlobalSetup so `spawn` can pick the right driver.
pub fn set_host(is_after_effects: bool) {
	HOST.store(if is_after_effects { HOST_AFTER_EFFECTS } else { HOST_PREMIERE }, Ordering::Relaxed);
}

fn drive_on_idle_pump() -> bool {
	HOST.load(Ordering::Relaxed) == HOST_AFTER_EFFECTS
}

/// Cooperative cancellation token, modeled on `tokio_util::CancellationToken`.
/// Clones share cancellation; [`child`](Self::child) makes a token also
/// cancelled when its parent is.
#[derive(Clone)]
pub struct CancelToken(Arc<CancelInner>);

struct CancelInner {
	cancelled: AtomicBool,
	parent:    Option<CancelToken>,
}

impl Default for CancelToken {
	fn default() -> Self {
		Self::new()
	}
}

impl CancelToken {
	pub fn new() -> Self {
		Self(Arc::new(CancelInner { cancelled: AtomicBool::new(false), parent: None }))
	}
	pub fn child(&self) -> CancelToken {
		Self(Arc::new(CancelInner { cancelled: AtomicBool::new(false), parent: Some(self.clone()) }))
	}
	pub fn cancel(&self) {
		self.0.cancelled.store(true, Ordering::SeqCst);
	}
	pub fn is_cancelled(&self) -> bool {
		self.0.cancelled.load(Ordering::SeqCst)
			|| self.0.parent.as_ref().is_some_and(|p| p.is_cancelled())
	}
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TaskId(pub u64);

#[derive(Clone, Debug)]
pub enum TaskStatus {
	Pending,
	Running { done: u32, total: u32 },
	Done,
	Cancelled,
	Failed(String),
}

/// Write side of a task's status, passed to [`BackgroundTask::poll`].
#[derive(Clone)]
pub struct StatusSink(Arc<Mutex<TaskStatus>>);

impl StatusSink {
	pub fn set(&self, status: TaskStatus) {
		*self.0.lock() = status;
		// A task progressed → UI may need to repaint (retained-mode).
		crate::effect::labels::mark_dirty();
	}
	pub fn progress(&self, done: u32, total: u32) {
		self.set(TaskStatus::Running { done, total });
	}
}

/// Read/cancel handle returned by [`spawn`]. Clone and stash anywhere; read
/// [`status`](Self::status) from a label expr to surface progress on any route.
#[derive(Clone)]
pub struct TaskHandle {
	pub id:    TaskId,
	pub token: CancelToken,
	status:    Arc<Mutex<TaskStatus>>,
}

impl TaskHandle {
	pub fn cancel(&self) {
		self.token.cancel();
	}
	pub fn status(&self) -> TaskStatus {
		self.status.lock().clone()
	}
}

/// Result of one [`BackgroundTask::poll`] tick.
pub enum TaskPoll {
	Pending,
	Done,
	Failed(String),
}

/// A unit of background work, polled on the main thread by the idle hook.
/// `poll` must not block; offload heavy/blocking work to a worker thread and
/// poll its result here, checking `token` to bail out early.
pub trait BackgroundTask: Send + 'static {
	fn poll(&mut self, token: &CancelToken, status: &StatusSink) -> TaskPoll;
}

struct TaskEntry {
	/// `Some` when the idle pump (AE) owns execution; `None` when a worker
	/// thread (Premiere) drives the task — the registry then only tracks the
	/// handle + tags for lookup/cancel.
	task:   Option<Box<dyn BackgroundTask>>,
	handle: TaskHandle,
	tags:   Vec<&'static str>,
}

fn registry() -> &'static Mutex<HashMap<TaskId, TaskEntry>> {
	static REG: OnceLock<Mutex<HashMap<TaskId, TaskEntry>>> = OnceLock::new();
	REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> TaskId {
	static COUNTER: AtomicU64 = AtomicU64::new(1);
	TaskId(COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Register a task, tagging it for later group lookup/cancellation. On AE it
/// runs on the next idle tick (via [`pump`]); on Premiere (or unknown host) it
/// runs immediately on a dedicated worker thread.
pub fn spawn<T: BackgroundTask>(task: T, tags: &[&'static str]) -> TaskHandle {
	let id = next_id();
	let handle = TaskHandle {
		id,
		token:  CancelToken::new(),
		status: Arc::new(Mutex::new(TaskStatus::Pending)),
	};

	if drive_on_idle_pump() {
		// AE: the idle hook owns execution; stash the task in the registry.
		registry().lock().insert(
			id,
			TaskEntry { task: Some(Box::new(task)), handle: handle.clone(), tags: tags.to_vec() },
		);
		log::info!("[tasks] spawned id={id:?} tags={tags:?} driver=idle-pump");
	} else {
		// Premiere / unknown: drive on a worker thread; registry tracks only
		// the handle + tags for lookup/cancel.
		registry().lock().insert(
			id,
			TaskEntry { task: None, handle: handle.clone(), tags: tags.to_vec() },
		);
		log::info!("[tasks] spawned id={id:?} tags={tags:?} driver=worker-thread");
		let token = handle.token.clone();
		let sink = StatusSink(handle.status.clone());
		let mut task = task;
		std::thread::spawn(move || {
			drive_to_completion(id, &mut task, &token, &sink);
		});
	}

	handle
}

/// Worker-thread driver (Premiere): poll the task to completion off the main
/// thread, sleeping between `Pending` ticks, bailing on cancellation. Removes
/// the registry entry when done.
fn drive_to_completion(
	id: TaskId,
	task: &mut dyn BackgroundTask,
	token: &CancelToken,
	sink: &StatusSink,
) {
	loop {
		if token.is_cancelled() {
			sink.set(TaskStatus::Cancelled);
			break;
		}
		match task.poll(token, sink) {
			TaskPoll::Pending => std::thread::sleep(Duration::from_millis(16)),
			TaskPoll::Done => {
				sink.set(TaskStatus::Done);
				break;
			}
			TaskPoll::Failed(e) => {
				sink.set(TaskStatus::Failed(e));
				break;
			}
		}
	}
	registry().lock().remove(&id);
}

/// Poll every registered task once. Called from the idle hook on the main
/// thread. Sets `max_sleep` (ms) low while work is pending so the next idle
/// tick comes soon, high when idle to avoid spinning.
pub fn pump(max_sleep: &mut i32) {
	// Confirm the host actually calls the idle hook (once), then per-tick only
	// when there's work — so the log isn't a 20 Hz flood.
	if !PUMP_SEEN.swap(true, Ordering::Relaxed) {
		log::info!("[tasks] idle hook fired for the first time — pump is live");
	}

	let mut reg = registry().lock();
	if !reg.is_empty() {
		log::info!("[tasks] pump: {} task(s) pending", reg.len());
	}
	let mut finished = Vec::new();

	for (id, entry) in reg.iter_mut() {
		let TaskEntry { task, handle, .. } = entry;
		// Worker-thread-driven tasks (None) are pumped by their thread, not here.
		let Some(task) = task else { continue };
		if handle.token.is_cancelled() {
			*handle.status.lock() = TaskStatus::Cancelled;
			finished.push(*id);
			continue;
		}
		let sink = StatusSink(handle.status.clone());
		match task.poll(&handle.token, &sink) {
			TaskPoll::Pending => {}
			TaskPoll::Done => {
				sink.set(TaskStatus::Done);
				finished.push(*id);
			}
			TaskPoll::Failed(e) => {
				sink.set(TaskStatus::Failed(e));
				finished.push(*id);
			}
		}
	}

	for id in finished {
		reg.remove(&id);
	}
	// Hint AE how soon to call us again: ~20 Hz while work is pending (ample
	// for wall-clock-paced tasks), idle otherwise.
	*max_sleep = if reg.is_empty() { 1000 } else { 50 };
}

pub fn get(id: TaskId) -> Option<TaskHandle> {
	registry().lock().get(&id).map(|e| e.handle.clone())
}

pub fn by_tag(tag: &'static str) -> Vec<TaskHandle> {
	registry().lock().values().filter(|e| e.tags.contains(&tag)).map(|e| e.handle.clone()).collect()
}

pub fn all() -> Vec<TaskHandle> {
	registry().lock().values().map(|e| e.handle.clone()).collect()
}

pub fn cancel(id: TaskId) {
	if let Some(h) = get(id) {
		h.cancel();
	}
}

pub fn cancel_tag(tag: &'static str) {
	for h in by_tag(tag) {
		h.cancel();
	}
}

pub fn cancel_all() {
	for h in all() {
		h.cancel();
	}
}
