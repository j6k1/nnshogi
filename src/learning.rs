use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;
use std::io::Write;

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
use nn::{Trainer};
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::{BufReader, Read, BufWriter};
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::path::Path;
use nncombinator::arr::Arr;
use nncombinator::layer::{BatchForwardBase, BatchTrain};
use nncombinator::persistence::{Persistence, TextFilePersistence};
use rand::prelude::SliceRandom;
use rand_xorshift::XorShiftRng;
use usiagent::output::USIStdErrorWriter;

#[derive(Debug,Deserialize,Serialize)]
pub struct CheckPoint {
	filename:String,
	item:usize
}
pub struct CheckPointReader {
	reader:BufReader<File>
}
impl CheckPointReader {
	pub fn new<P: AsRef<Path>>(file:P) -> Result<CheckPointReader,ApplicationError> {
		if file.as_ref().exists() {
			Ok(CheckPointReader {
				reader: BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)
			})
		} else {
			Err(ApplicationError::StartupError(String::from(
				"指定されたチェックポイントファイルは存在しません。"
			)))
		}
	}
	pub fn read(&mut self) -> Result<CheckPoint,ApplicationError> {
		let mut buf = String::new();
		self.reader.read_to_string(&mut buf)?;
		match toml::from_str(buf.as_str()) {
			Ok(r) => Ok(r),
			Err(ref e) => {
				let _ = USIStdErrorWriter::write(&e.to_string());
				Err(ApplicationError::StartupError(String::from(
					"チェックポイントファイルのロード時にエラーが発生しました。"
				)))
			}
		}
	}
}
pub struct CheckPointWriter<P: AsRef<Path>> {
	writer:BufWriter<File>,
	tmp:P,
	path:P
}
impl<P: AsRef<Path>> CheckPointWriter<P> {
	pub fn new(tmp:P,file:P) -> Result<CheckPointWriter<P>,ApplicationError> {
		Ok(CheckPointWriter {
			writer: BufWriter::new(OpenOptions::new().write(true).create(true).open(&tmp)?),
			tmp:tmp,
			path:file
		})
	}
	pub fn save(&mut self,checkpoint:&CheckPoint) -> Result<(),ApplicationError> {
		let toml_str = toml::to_string(checkpoint)?;

		match write!(self.writer,"{}",toml_str) {
			Ok(()) => {
				self.writer.flush()?;
				fs::rename(&self.tmp,&self.path)?;
				Ok(())
			},
			Err(_) => {
				Err(ApplicationError::StartupError(String::from(
					"チェックポイントファイルの保存時にエラーが発生しました。"
				)))
			}
		}
	}
}
pub struct Learnener<NN>
	where NN: BatchForwardBase<BatchInput=Vec<Arr<f32,2517>>,BatchOutput=Vec<Arr<f32,1>>> +
			  BatchTrain<f32> + Persistence<f32,TextFilePersistence<f32>> {
	nn:PhantomData<NN>,
	bias_shake_shake:bool
}
impl<NN> Learnener<NN>
	where NN: BatchForwardBase<BatchInput=Vec<Arr<f32,2517>>,BatchOutput=Vec<Arr<f32,1>>> +
			  BatchTrain<f32> + Persistence<f32,TextFilePersistence<f32>>{
	pub fn new(bias_shake_shake:bool) -> Learnener<NN> {
		Learnener {
			nn:PhantomData::<NN>,
			bias_shake_shake:bias_shake_shake
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

	pub fn learning_from_csa(&mut self, kifudir:String, lowerrate:f64, evalutor: Trainer<NN>,bias_shake_shake:bool, learn_max_threads:usize) -> Result<(),ApplicationError> {
		let logger = FileLogger::new(String::from("logs/log.txt"))?;

		let logger = Arc::new(Mutex::new(logger));
		let on_error_handler_arc = Arc::new(Mutex::new(OnErrorHandler::new(logger.clone())));

		let system_event_queue_arc:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));
		let user_event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));

		let notify_quit_arc = Arc::new(AtomicBool::new(false));

		let on_error_handler = on_error_handler_arc.clone();

		let notify_quit = notify_quit_arc.clone();

		let mut system_event_dispatcher = self.create_event_dispatcher(notify_quit,on_error_handler);

		let mut evalutor = evalutor;

		print!("learning start... kifudir = {}\n", kifudir);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();

		self.start_read_stdinput_thread(system_event_queue,on_error_handler);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();
		let notify_quit = notify_quit_arc.clone();

		let mut count = 0;

		let checkpoint_path = Path::new(&kifudir).join("checkpoint.toml");

		let checkpoint = if checkpoint_path.exists() {
			Some(CheckPointReader::new(&checkpoint_path)?.read()?)
		} else {
			None
		};

		let mut current_item:usize;
		let mut current_filename;

		let mut skip_files = checkpoint.is_some();
		let mut skip_items = checkpoint.is_some();

		'files: for entry in fs::read_dir(kifudir)? {
			let path = entry?.path();

			current_filename = Some(path.as_path().file_name().map(|s| {
				s.to_string_lossy().to_string()
			}).unwrap_or(String::from("")));

			if let Some(ref checkpoint) = checkpoint {
				if *current_filename.as_ref().unwrap() == checkpoint.filename {
					skip_files = false;
				}

				if skip_files {
					continue;
				}
			}

			if !path.as_path().extension().map(|e| e == "csa").unwrap_or(false) {
				continue;
			}

			print!("{}\n", path.display());
			let parsed:Vec<CsaData> = CsaParser::new(CsaFileStream::new(path)?).parse()?;

			current_item = 0;

			for p in parsed.into_iter() {
				if let Some(ref checkpoint) = checkpoint {
					if skip_items && current_item == checkpoint.item {
						println!("Processing starts from {}th item of file {}",current_item,current_filename.as_ref().unwrap());
						skip_items = false;
					}

					if skip_items {
						continue;
					}
				}
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

					let a: f32 = rnd.gen();
					let b: f32 = 1f32 - a;

					(a,b)
				} else {
					(0.5f32,0.5f32)
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
							0f32
						},
						&GameEndState::Lose if t == teban => {
							0f32
						},
						&GameEndState::Lose => {
							ab
						},
						_ => 0.5f32
					}
				}, a,b, &*user_event_queue) {
					Err(_) => {
						return Err(ApplicationError::LearningError(String::from(
							"An error occurred while learning the neural network."
						)));
					},
					Ok((msa,moa,msb,mob)) => {
						println!("error_total: {}, {}, {}, {}",msa, moa, msb, mob);
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

			let tmp_path = format!("{}.tmp",checkpoint_path.as_path().to_string_lossy());
			let tmp_path = Path::new(&tmp_path);

			let mut checkpoint_writer = CheckPointWriter::new(tmp_path,checkpoint_path.as_path())?;

			checkpoint_writer.save(&CheckPoint {
				filename:current_filename.as_ref().unwrap().clone(),
				item:current_item
			})?;
		}

		print!("{}件の棋譜を学習しました。\n",count);

		Ok(())
	}

	pub fn learning_from_yaneuraou_bin(&mut self, kifudir:String,
									   evalutor: Trainer<NN>,
									   bias_shake_shake:bool,
									   learn_max_threads:usize,
									   learn_sfen_read_size:usize,
									   learn_batch_size:usize,
									   ) -> Result<(),ApplicationError> {
		self.learning_batch(kifudir,
							"bin",
							evalutor,
							bias_shake_shake,
							learn_max_threads,
							learn_sfen_read_size,
							learn_batch_size,
							Self::learning_from_yaneuraou_bin_batch)

	}

	pub fn learning_from_hcpe(&mut self, kifudir:String,
									   evalutor: Trainer<NN>,
									   bias_shake_shake:bool,
									   learn_max_threads:usize,
									   learn_sfen_read_size:usize,
									   learn_batch_size:usize,
	) -> Result<(),ApplicationError> {
		self.learning_batch(kifudir,
							"hcpe",
							evalutor,
							bias_shake_shake,
							learn_max_threads,
							learn_sfen_read_size,
							learn_batch_size,
							Self::learning_from_hcpe_batch)

	}

	pub fn learning_batch(&mut self,kifudir:String,
							   ext:&str,
							   evalutor: Trainer<NN>,
							   bias_shake_shake:bool,
							   learn_max_threads:usize,
							   learn_sfen_read_size:usize,
							   learn_batch_size:usize,
							   learning_process:fn(
								   &mut Trainer<NN>,
								   Vec<Vec<u8>>,
								   bool,
								   usize,
								   &Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
							   ) -> Result<(),ApplicationError>
	) -> Result<(),ApplicationError> {
		let logger = FileLogger::new(String::from("logs/log.txt"))?;

		let logger = Arc::new(Mutex::new(logger));
		let on_error_handler_arc = Arc::new(Mutex::new(OnErrorHandler::new(logger.clone())));

		let system_event_queue_arc:Arc<Mutex<EventQueue<SystemEvent,SystemEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));
		let user_event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>> = Arc::new(Mutex::new(EventQueue::new()));

		let notify_quit_arc = Arc::new(AtomicBool::new(false));

		let on_error_handler = on_error_handler_arc.clone();

		let notify_quit = notify_quit_arc.clone();

		let mut system_event_dispatcher = self.create_event_dispatcher(notify_quit,on_error_handler);

		let mut evalutor = evalutor;

		print!("learning start... kifudir = {}\n", kifudir);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();

		self.start_read_stdinput_thread(system_event_queue,on_error_handler);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();
		let notify_quit = notify_quit_arc.clone();

		let mut count = 0;

		let mut teachers = Vec::with_capacity(learn_sfen_read_size);
		let mut record = Vec::with_capacity(40);

		let checkpoint_path = Path::new(&kifudir).join("checkpoint.toml");

		let checkpoint = if checkpoint_path.exists() {
			Some(CheckPointReader::new(&checkpoint_path)?.read()?)
		} else {
			None
		};

		let mut current_item:usize = 0;
		let mut current_filename = None;

		let mut skip_files = checkpoint.is_some();
		let mut skip_items = checkpoint.is_some();

		'files: for entry in fs::read_dir(kifudir)? {
			let path = entry?.path();

			current_filename = Some(path.as_path().file_name().map(|s| {
				s.to_string_lossy().to_string()
			}).unwrap_or(String::from("")));

			if let Some(ref checkpoint) = checkpoint {
				if *current_filename.as_ref().unwrap() == checkpoint.filename {
					skip_files = false;
				}

				if skip_files {
					continue;
				}
			}

			if !path.as_path().extension().map(|e| e == ext).unwrap_or(false) {
				continue;
			}

			print!("{}\n", path.display());

			current_item = 0;

			for b in BufReader::new(File::open(path)?).bytes() {
				let b = b?;

				record.push(b);

				if record.len() == 40 {
					if let Some(ref checkpoint) = checkpoint {
						if skip_items && current_item < checkpoint.item {
							current_item += 1;
							record.clear();
							continue;
						} else {
							if skip_items && current_item == checkpoint.item {
								println!("Processing starts from {}th item of file {}",current_item,current_filename.as_ref().unwrap());
								skip_items = false;
							}
						}
					}
					teachers.push(record);
					record = Vec::with_capacity(40);
				} else {
					continue;
				}

				if teachers.len() == learn_sfen_read_size {
					let mut rng = rand::thread_rng();
					teachers.shuffle(&mut rng);

					let mut batch = Vec::with_capacity(learn_batch_size);

					let it = teachers.into_iter();
					teachers = Vec::with_capacity(learn_sfen_read_size);

					for sfen in it {
						batch.push(sfen);

						if batch.len() == learn_batch_size {
							learning_process(&mut evalutor,
													batch,
													bias_shake_shake,
													learn_max_threads,
													&user_event_queue)?;
							batch = Vec::with_capacity(learn_batch_size);
							count += learn_batch_size;
							current_item += learn_batch_size;

							let tmp_path = format!("{}.tmp",&checkpoint_path.as_path().to_string_lossy());
							let tmp_path = Path::new(&tmp_path);

							let mut checkpoint_writer = CheckPointWriter::new(tmp_path,&checkpoint_path.as_path())?;

							checkpoint_writer.save(&CheckPoint {
								filename:current_filename.as_ref().unwrap().clone(),
								item:current_item
							})?;
						}

						if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
							let _ = on_error_handler.lock().map(|h| h.call(e));
						}

						if notify_quit.load(Ordering::Acquire) {
							break 'files;
						}
					}

					let remaing = batch.len();

					if remaing > 0 {
						learning_process(&mut evalutor,
												batch,
												bias_shake_shake,
												learn_max_threads,
												&user_event_queue)?;
						count += remaing;
					}

					if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
						let _ = on_error_handler.lock().map(|h| h.call(e));
					}

					if notify_quit.load(Ordering::Acquire) {
						break 'files;
					}
				}
			}
		}

		if record.len() > 0 {
			return Err(ApplicationError::LearningError(String::from(
				"The data size of the teacher phase is invalid."
			)));
		}

		if !notify_quit.load(Ordering::Acquire) && teachers.len() > 0 {
			let mut rng = rand::thread_rng();
			teachers.shuffle(&mut rng);

			let mut batch = Vec::with_capacity(learn_batch_size);

			for sfen in teachers.into_iter() {
				let (a,b) = if self.bias_shake_shake {
					let mut rnd = rand::thread_rng();
					let mut rnd = XorShiftRng::from_seed(rnd.gen());

					let a = rnd.gen();
					let b = 1f32 - a;

					(a,b)
				} else {
					if self.bias_shake_shake {
						let mut rnd = rand::thread_rng();
						let mut rnd = XorShiftRng::from_seed(rnd.gen());

						let a = rnd.gen();
						let b = 1f32 - a;

						(a,b)
					} else {
						(0.5f32,0.5f32)
					}
				};

				if batch.len() == learn_batch_size {
					learning_process(&mut evalutor,
											batch,
											bias_shake_shake,
											learn_max_threads,
											&user_event_queue)?;
					batch = Vec::with_capacity(learn_batch_size);
					count += learn_batch_size;
					current_item += learn_batch_size;

					let tmp_path = format!("{}.tmp",&checkpoint_path.as_path().to_string_lossy());
					let tmp_path = Path::new(&tmp_path);

					let mut checkpoint_writer = CheckPointWriter::new(tmp_path,&checkpoint_path.as_path())?;

					checkpoint_writer.save(&CheckPoint {
						filename:current_filename.as_ref().unwrap().clone(),
						item:current_item
					})?;
				}


				if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
					let _ = on_error_handler.lock().map(|h| h.call(e));
				}

				if notify_quit.load(Ordering::Acquire) {
					print!("{}局面を学習しました。\n",count);
					return Ok(());
				}
			}

			let remaing = batch.len();

			if !notify_quit.load(Ordering::Acquire) && remaing > 0 {
				learning_process(&mut evalutor,
							   		batch,
							   		bias_shake_shake,
							   		learn_max_threads,
							   		&user_event_queue)?;
				count += remaing;
			}
		}

		print!("{}局面を学習しました。\n",count);

		Ok(())
	}

	fn learning_from_yaneuraou_bin_batch(evalutor:&mut Trainer<NN>,
										 batch:Vec<Vec<u8>>,
										 bias_shake_shake:bool,
										 learn_max_threads:usize,
										 user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
	) -> Result<(),ApplicationError> {

		let (a,b) = if bias_shake_shake {
			let mut rnd = rand::thread_rng();

			let a: f32 = rnd.gen();
			let b: f32 = 1f32 - a;

			(a,b)
		} else {
			(1f32,1f32)
		};

		match evalutor.learning_by_packed_sfens(
			batch,
			learn_max_threads,
			&move |s, ab| {

				match s {
					&GameEndState::Win => {
						ab
					}
					&GameEndState::Lose => {
						0f32
					},
					_ => 0.5f32
				}
			}, a,b, &*user_event_queue) {
			Err(_) => {
				return Err(ApplicationError::LearningError(String::from(
					"An error occurred while learning the neural network."
				)));
			},
			Ok((msa,moa,msb,mob)) => {
				println!("error_total: {}, {}, {}, {}",msa, moa, msb, mob);
				Ok(())
			}
		}
	}

	fn learning_from_hcpe_batch(evalutor: &mut Trainer<NN>,
								batch:Vec<Vec<u8>>,
								bias_shake_shake:bool,
								learn_max_threads:usize,
								user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
	) -> Result<(),ApplicationError> {

			let (a,b) = if bias_shake_shake {
			let mut rnd = rand::thread_rng();

			let a: f32 = rnd.gen();
			let b: f32 = 1f32 - a;

			(a,b)
		} else {
			(1f32,1f32)
		};

		match evalutor.learning_by_hcpe(
			batch,
			learn_max_threads,
			&move |s, ab| {

				match s {
					&GameEndState::Win => {
						ab
					}
					&GameEndState::Lose => {
						0f32
					},
					_ => 0.5f32
				}
			}, a,b, &*user_event_queue) {
			Err(_) => {
				return Err(ApplicationError::LearningError(String::from(
					"An error occurred while learning the neural network."
				)));
			},
			Ok((msa,moa,msb,mob)) => {
				println!("error_total: {}, {}, {}, {}",msa, moa, msb, mob);
				Ok(())
			}
		}
	}
}