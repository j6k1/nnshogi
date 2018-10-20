use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;

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

pub struct CsaLearnener {

}
impl CsaLearnener {
	pub fn new() -> CsaLearnener {
		CsaLearnener {

		}
	}

	pub fn learning(&mut self,kifudir:String,lowerrate:f64) -> Result<(),ApplicationError> {
		let logger = FileLogger::new(String::from("logs/log.txt"))?;

		let logger = Arc::new(Mutex::new(logger));
		let on_error_handler_arc = Arc::new(Mutex::new(OnErrorHandler::new(logger.clone())));

		let system_event_queue_arc:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));
		let user_event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));

		let mut system_event_dispatcher:USIEventDispatcher<SystemEventKind,
														SystemEvent,(),FileLogger,CommonError> = USIEventDispatcher::new(&logger);

		let notify_quit_arc = Arc::new(Mutex::new(false));

		let on_error_handler = on_error_handler_arc.clone();

		let notify_quit = notify_quit_arc.clone();

		system_event_dispatcher.add_handler(SystemEventKind::Quit, move |_,e| {
			match e {
				&SystemEvent::Quit => {
					match notify_quit.lock() {
						Ok(mut notify_quit) => {
							*notify_quit = true;
						},
						Err(ref e) => {
							on_error_handler.lock().map(|h| h.call(e)).is_err();
						}
					};
					Ok(())
				},
				e => Err(EventHandlerError::InvalidState(e.event_kind())),
			}
		});

		let mut evalutor = Intelligence::new(String::from("data"),
															String::from("nn.a.bin"),
															String::from("nn.b.bin"),true);

		print!("learning start... kifudir = {}\n", kifudir);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();

		thread::spawn(move || {
			let mut input_reader = USIStdInputReader::new();

			loop {
				match input_reader.read() {
					Ok(line) => {
						match line.trim_right() {
							"quit" => {
								match system_event_queue.lock() {
									Ok(mut system_event_queue) => {
										system_event_queue.push(SystemEvent::Quit);
									},
									Err(ref e) => {
										on_error_handler.lock().map(|h| h.call(e)).is_err();
										return;
									}
								}
							},
							_ => (),
						}
					},
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
						match system_event_queue.lock() {
							Ok(mut system_event_queue) => {
								system_event_queue.push(SystemEvent::Quit);
							},
							Err(ref e) => {
								on_error_handler.lock().map(|h| h.call(e)).is_err();
								return;
							}
						}
						return;
					}
				}
			}
		});

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();
		let notify_quit = notify_quit_arc.clone();

		let mut count = 0;

		'files: for entry in fs::read_dir(kifudir)? {
			let path = entry?.path();
			print!("{}\n", path.display());
			let parsed:Vec<CsaData> = CsaParser::new(CsaFileStream::new(path)?).parse()?;

			for p in parsed.into_iter() {
				match p.end_state {
					Some(EndState::Toryo) | Some(EndState::Tsumi) => (),
					_ => {
						continue;
					}
				}

				if !p.comments.iter().any(|ref mut c| {
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
				let teban_at_start = teban;
				let banmen = p.initial_position;
				let mc = p.initial_mochigoma;

				let history:Vec<(Banmen,MochigomaCollections,u64,u64)> = Vec::new();

				let (teban,_,_,history) = Rule::apply_moves_with_callback(&banmen,
																teban,
																mc,&m,history,
																|banmen,_,mc,_,_,history| {
					let mut history = history;
					history.push((banmen.clone(),mc.clone(),0,0));
					history
				});

				let s = match p.end_state {
					Some(EndState::Toryo) if teban == teban_at_start => GameEndState::Win,
					Some(EndState::Toryo) => GameEndState::Lose,
					Some(EndState::Tsumi) if teban == teban_at_start => GameEndState::Lose,
					Some(EndState::Tsumi) => GameEndState::Win,
					_ => {
						return Err(ApplicationError::LogicError(String::from(
							"current EndState invalid!"
						)));
					}
				};

				match evalutor.learning(teban_at_start,teban,history,&s,&*user_event_queue) {
					Err(_) => {
						return Err(ApplicationError::LearningError(String::from(
							"An error occurred while learning the neural network."
						)));
					},
					_ => (),
				}

				count += 1;

				if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
				}

				match notify_quit.lock() {
					Ok(mut notify_quit) => {
						if *notify_quit {
							break 'files;
						}
					},
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
						return Err(ApplicationError::LearningError(String::from(
							"End notification flag's exclusive lock could not be secured"
						)));
					}
				};
			}

			print!("done... \n");
		}

		print!("{}件の棋譜を学習しました。\n",count);

		Ok(())
	}
}