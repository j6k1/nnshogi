use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;

use rand::seq::SliceRandom;

use usiagent::OnErrorHandler;
use usiagent::shogi::Banmen;
use usiagent::shogi::MochigomaCollections;
use usiagent::rule::*;
use usiagent::event::*;
use usiagent::logger::*;
use usiagent::input::*;
use usiagent::error::*;

use csaparser::CsaParser;
use csaparser::CsaFileStream;
use csaparser::CsaMove;
use csaparser::CsaData;
use csaparser::EndState;

use error::ApplicationError;
use error::CommonError;
use nn::Intelligence;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct CsaLearnener {

}
impl CsaLearnener {
	pub fn new() -> CsaLearnener {
		CsaLearnener {

		}
	}

	fn create_event_dispatcher(&self, notify_quit:Arc<AtomicBool>,
							   		  on_error_handler:Arc<Mutex<OnErrorHandler<FileLogger>>>) -> USIEventDispatcher<SystemEventKind,
		SystemEvent,(),FileLogger,CommonError> {

		let mut system_event_dispatcher:USIEventDispatcher<SystemEventKind,
			SystemEvent,(),FileLogger,CommonError> = USIEventDispatcher::new(&on_error_handler);

		system_event_dispatcher.add_handler(SystemEventKind::Quit, move |_,e| {
			match e {
				&SystemEvent::Quit => {
					notify_quit.store(true,Ordering::Release);
					Ok(())
				},
				e => Err(EventHandlerError::InvalidState(e.event_kind())),
			}
		});

		system_event_dispatcher
	}

	fn start_read_stdinput_thread(&self,
								  system_event_queue:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>>,
								  on_error_handler:Arc<Mutex<OnErrorHandler<FileLogger>>>) {
		thread::spawn(move || {
			let mut input_reader = USIStdInputReader::new();

			loop {
				match input_reader.read() {
					Ok(line) => {
						match line.trim_end() {
							"quit" => {
								match system_event_queue.lock() {
									Ok(mut system_event_queue) => {
										system_event_queue.push(SystemEvent::Quit);
										return;
									},
									Err(ref e) => {
										let _ = on_error_handler.lock().map(|h| h.call(e));
										return;
									}
								}
							},
							_ => (),
						}
					},
					Err(ref e) => {
						let _ = on_error_handler.lock().map(|h| h.call(e));
						match system_event_queue.lock() {
							Ok(mut system_event_queue) => {
								system_event_queue.push(SystemEvent::Quit);
							},
							Err(ref e) => {
								let _ = on_error_handler.lock().map(|h| h.call(e));
							}
						}
						return;
					}
				}
			}
		});
	}

	pub fn learning_from_csa(&mut self, kifudir:String, lowerrate:f64, bias_shake_shake:bool, learn_max_threads:usize) -> Result<(),ApplicationError> {
		let logger = FileLogger::new(String::from("logs/log.txt"))?;

		let logger = Arc::new(Mutex::new(logger));
		let on_error_handler_arc = Arc::new(Mutex::new(OnErrorHandler::new(logger.clone())));

		let system_event_queue_arc:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));
		let user_event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));

		let notify_quit_arc = Arc::new(AtomicBool::new(false));

		let on_error_handler = on_error_handler_arc.clone();

		let notify_quit = notify_quit_arc.clone();

		let mut system_event_dispatcher = self.create_event_dispatcher(notify_quit,on_error_handler);

		let mut evalutor = Intelligence::new(String::from("data"),
															String::from("nn.a.bin"),
															String::from("nn.b.bin"),false);

		print!("learning start... kifudir = {}\n", kifudir);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();

		self.start_read_stdinput_thread(system_event_queue,on_error_handler);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();
		let notify_quit = notify_quit_arc.clone();

		let mut count = 0;

		'files: for entry in fs::read_dir(kifudir)? {
			let path = entry?.path();

			if !path.as_path().extension().map(|e| e == "csa").unwrap_or(false) {
				continue;
			}

			print!("{}\n", path.display());
			let parsed:Vec<CsaData> = CsaParser::new(CsaFileStream::new(path)?).parse()?;

			for p in parsed.into_iter() {
				match p.end_state {
					Some(EndState::Toryo) | Some(EndState::Tsumi) => (),
					_ => {
						continue;
					}
				}

				if !p.comments.iter().any(|c| {
					if !c.starts_with("white_rate:") && !c.starts_with("black_rate:") {
						return false;
					}

					let c = c.split(':').collect::<Vec<&str>>();

					if c.len() != 3 {
						false
					} else {
						let rate:f64 = match c[2].parse() {
							Err(_) => {
								return false;
							},
							Ok(rate) => rate,
						};

						rate >= lowerrate
					}
				}) {
					continue;
				}
				let m = p.moves.iter().fold(Vec::new(),|mut mvs,m| match *m {
					CsaMove::Move(m,_) => {
						mvs.push(m);
						mvs
					},
					_ => {
						mvs
					}
				});
				let teban = p.teban_at_start;
				let banmen = p.initial_position;
				let state = State::new(banmen);
				let mc = p.initial_mochigoma;

				let history:Vec<(Banmen,MochigomaCollections,u64,u64)> = Vec::new();

				let (teban,_,_,history) = Rule::apply_moves_with_callback(state,
																teban,
																mc,&m.into_iter().map(|m| {
																	m.to_applied_move()
																}).collect::<Vec<AppliedMove>>(),
																history,
																|_,banmen,mc,_,_,history| {
					let mut history = history;
					history.push((banmen.clone(),mc.clone(),0,0));
					history
				});

				let (a,b) = if bias_shake_shake {
					let mut rnd = rand::thread_rng();

					let a: f64 = rnd.gen();
					let b: f64 = 1f64 - a;

					(a,b)
				} else {
					(1f64,1f64)
				};

				let teban = teban.opposite();

				match evalutor.learning_by_training_data(
					teban,
					history,
					&GameEndState::Win,
					learn_max_threads,
					&move |s,t, ab| {

					match s {
						&GameEndState::Win if t == teban => {
							ab
						}
						&GameEndState::Win => {
							0f64
						},
						&GameEndState::Lose if t == teban => {
							0f64
						},
						&GameEndState::Lose => {
							ab
						},
						_ => 0.5f64
					}
				}, a,b, &*user_event_queue) {
					Err(_) => {
						return Err(ApplicationError::LearningError(String::from(
							"An error occurred while learning the neural network."
						)));
					},
					Ok((msa,moa,msb,mob)) => {
						println!("error_total: {}, error_average: {}",msa.error_total + moa.error_total,(msa.error_average + moa.error_average) / 2f64);
						println!("error_total: {}, error_average: {}",msb.error_total + mob.error_total,(msb.error_average + mob.error_average) / 2f64);
					}
				};

				count += 1;

				if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
					let _ = on_error_handler.lock().map(|h| h.call(e));
				}

				if notify_quit.load(Ordering::Acquire) {
					break 'files;
				}
			}

			print!("done... \n");
		}

		print!("{}件の棋譜を学習しました。\n",count);

		Ok(())
	}
}