use std::thread;
use std::sync::Mutex;
use std::sync::Arc;
use std::fs;
use std::io::Write;

use usiagent::OnErrorHandler;
use usiagent::shogi::{Banmen, Teban};
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::io::{BufReader, Read, BufWriter};
use std::fs::{DirEntry, File, OpenOptions};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use nncombinator::arr::{Arr, VecArr};
use nncombinator::device::DeviceCpu;
use nncombinator::layer::{BatchForwardBase, BatchTrain, ForwardAll};
use nncombinator::persistence::{BinFilePersistence, Linear, Persistence};
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
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
	where NN: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
			  BatchForwardBase<BatchInput=VecArr<f32,Arr<f32,2517>>,BatchOutput=VecArr<f32,Arr<f32,1>>> +
			  BatchTrain<f32,DeviceCpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear> {
	nn:PhantomData<NN>}
impl<NN> Learnener<NN>
	where NN: ForwardAll<Input=Arr<f32,2517>,Output=Arr<f32,1>> +
			  BatchForwardBase<BatchInput=VecArr<f32,Arr<f32,2517>>,BatchOutput=VecArr<f32,Arr<f32,1>>> +
			  BatchTrain<f32,DeviceCpu<f32>> + Persistence<f32,BinFilePersistence<f32>,Linear>{
	pub fn new() -> Learnener<NN> {
		Learnener {
			nn:PhantomData::<NN>
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

	fn start_read_stdinput_thread(&self,notify_run_test:Arc<AtomicBool>,
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
										notify_run_test.store(false,Ordering::Release);
										system_event_queue.push(SystemEvent::Quit);
										return;
									},
									Err(ref e) => {
										let _ = on_error_handler.lock().map(|h| h.call(e));
										return;
									}
								}
							},
							"test" => {
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
							}
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

	pub fn learning_from_csa(&mut self, kifudir:String, lowerrate:f64, evalutor: Trainer<NN>) -> Result<(),ApplicationError> {
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

		let notify_run_test_arc = Arc::new(AtomicBool::new(true));
		let notify_run_test = notify_run_test_arc.clone();

		self.start_read_stdinput_thread(notify_run_test,system_event_queue,on_error_handler);

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

		let mut rng = rand::thread_rng();
		let mut rng = XorShiftRng::from_seed(rng.gen());

		let mut paths = fs::read_dir(Path::new(&kifudir)
										.join("training"))?.into_iter()
										.collect::<Vec<Result<DirEntry,_>>>();

		paths.sort_by(|a,b| {
			match (a,b) {
				(Ok(a),Ok(b)) => {
					let a = a.file_name();
					let b = b.file_name();
					a.cmp(&b)
				},
				_ => {
					std::cmp::Ordering::Equal
				}
			}
		});

		'files: for path in paths {
			let path = path?.path();

			current_filename = path.as_path().file_name().map(|s| {
				s.to_string_lossy().to_string()
			}).unwrap_or(String::from(""));

			if let Some(ref checkpoint) = checkpoint {
				if current_filename == checkpoint.filename {
					skip_files = false;
				}

				if skip_files {
					continue;
				} else if current_filename != checkpoint.filename {
					skip_items = false;
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
						println!("Processing starts from {}th item of file {}",current_item,&current_filename);
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

				let teban = teban.opposite();

				match evalutor.learning_by_training_csa(
					teban,
					history,
					&GameEndState::Win,&*user_event_queue) {
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
				filename:current_filename.clone(),
				item:current_item
			})?;
		}

		if notify_run_test_arc.load(Ordering::Acquire) {
			let mut historys:Vec<(Teban,GameEndState,(Banmen,MochigomaCollections,u64,u64))> = Vec::new();


			let mut paths = fs::read_dir(Path::new(&kifudir)
				.join("tests"))?.into_iter()
				.collect::<Vec<Result<DirEntry,_>>>();

			paths.sort_by(|a,b| {
				match (a,b) {
					(Ok(a),Ok(b)) => {
						let a = a.file_name();
						let b = b.file_name();
						a.cmp(&b)
					},
					_ => {
						std::cmp::Ordering::Equal
					}
				}
			});

			'test_files: for path in paths {
				let path = path?.path();

				if !path.as_path().extension().map(|e| e == "csa").unwrap_or(false) {
					continue;
				}

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
					let history = Vec::new();

					let (mut teban,_,_,history) = Rule::apply_moves_with_callback(state,
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

					let mut s = GameEndState::Win;

					for h in history.into_iter() {
						historys.push((teban,s.clone(),h));
						teban = teban.opposite();

						if s == GameEndState::Win {
							s = GameEndState::Lose;
						} else {
							s = GameEndState::Win;
						}
					}

					if historys.len() >= 10000 {
						break 'test_files;
					}
				}
			}

			historys.shuffle(&mut rng);

			let mut successed = 0;
			let mut count = 0;

			for (teban,s,kyokumen) in historys.iter().take(100) {
				let score = evalutor.test_by_csa(*teban, kyokumen)?;

				let success = match s {
					GameEndState::Win => {
						score >= 0.5
					},
					_ => {
						score < 0.5
					}
				};

				if success {
					successed += 1;
					println!("勝率{}% 正解!",score * 100.);
				} else {
					println!("勝率{}% 不正解...",score * 100.);
				}

				count += 1;
			}

			println!("正解率 {}%",successed as f32 / count as f32 * 100.);
		}

		print!("{}件の棋譜を学習しました。\n",count);

		Ok(())
	}

	pub fn learning_from_yaneuraou_bin(&mut self, kifudir:String,
									   evalutor: Trainer<NN>,
									   learn_sfen_read_size:usize,
									   learn_batch_size:usize,
									   save_batch_count:usize,
									   ) -> Result<(),ApplicationError> {
		self.learning_batch(kifudir,
							"bin",
							40,
							evalutor,
							learn_sfen_read_size,
							learn_batch_size,
							save_batch_count,
							Self::learning_from_yaneuraou_bin_batch,
							|evalutor,packed| {
								evalutor.test_by_packed_sfens(packed)
							})

	}

	pub fn learning_from_hcpe(&mut self, kifudir:String,
									   evalutor: Trainer<NN>,
									   learn_sfen_read_size:usize,
									   learn_batch_size:usize,
							  		   save_batch_count:usize,
	) -> Result<(),ApplicationError> {
		self.learning_batch(kifudir,
							"hcpe",
								38,
							evalutor,
							learn_sfen_read_size,
							learn_batch_size,
							save_batch_count,
							Self::learning_from_hcpe_batch,
							|evalutor,packed| {
								evalutor.test_by_packed_hcpe(packed)
							})

	}

	pub fn learning_batch<F>(&mut self,kifudir:String,
							   ext:&str,
							   item_size:usize,
							   evalutor: Trainer<NN>,
							   learn_sfen_read_size:usize,
							   learn_batch_size:usize,
							   save_batch_count:usize,
							   learning_process:fn(
								   &mut Trainer<NN>,
								   Vec<Vec<u8>>,
								   &Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
							   ) -> Result<(),ApplicationError>,
							   mut test_process:F
	) -> Result<(),ApplicationError>
		where F: FnMut(&mut Trainer<NN>,Vec<u8>) -> Result<(GameEndState,f32),ApplicationError> {

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

		let notify_run_test_arc = Arc::new(AtomicBool::new(true));
		let notify_run_test = notify_run_test_arc.clone();

		self.start_read_stdinput_thread(notify_run_test,system_event_queue,on_error_handler);

		let on_error_handler = on_error_handler_arc.clone();
		let system_event_queue = system_event_queue_arc.clone();
		let notify_quit = notify_quit_arc.clone();

		let mut count = 0;

		let mut teachers = Vec::with_capacity(learn_sfen_read_size);
		let mut record = Vec::with_capacity(item_size);

		let mut pending_count = 0;

		let checkpoint_path = Path::new(&kifudir).join("checkpoint.toml");

		let checkpoint = if checkpoint_path.exists() {
			Some(CheckPointReader::new(&checkpoint_path)?.read()?)
		} else {
			None
		};

		let mut current_item:usize = 0;
		let mut current_filename = String::from("");

		let mut skip_files = checkpoint.is_some();
		let mut skip_items = checkpoint.is_some();

		let mut rng = rand::thread_rng();
		let mut rng = XorShiftRng::from_seed(rng.gen());

		let mut paths = fs::read_dir(Path::new(&kifudir)
			.join("training"))?.into_iter()
			.collect::<Vec<Result<DirEntry,_>>>();

		paths.sort_by(|a,b| {
			match (a,b) {
				(Ok(a),Ok(b)) => {
					let a = a.file_name();
					let b = b.file_name();
					a.cmp(&b)
				},
				_ => {
					std::cmp::Ordering::Equal
				}
			}
		});

		'files: for path in paths {
			let path = path?.path();

			current_filename = path.as_path().file_name().map(|s| {
				s.to_string_lossy().to_string()
			}).unwrap_or(String::from(""));

			if let Some(ref checkpoint) = checkpoint {
				if current_filename == checkpoint.filename {
					skip_files = false;
				}

				if skip_files {
					continue;
				} else if current_filename != checkpoint.filename {
					skip_items = false;
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

				if record.len() == item_size {
					if let Some(ref checkpoint) = checkpoint {
						if skip_items && current_item < checkpoint.item {
							current_item += 1;
							record.clear();
							continue;
						} else {
							if skip_items && current_item == checkpoint.item {
								println!("Processing starts from {}th item of file {}",current_item,&current_filename);
								skip_items = false;
							}
						}
					}
					teachers.push(record);
					record = Vec::with_capacity(item_size);
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
													&user_event_queue)?;
							pending_count += 1;

							batch = Vec::with_capacity(learn_batch_size);
							count += learn_batch_size;
							current_item += learn_batch_size;

							if pending_count >= save_batch_count {
								self.save(&mut evalutor,&checkpoint_path,&current_filename,current_item)?;
								pending_count = 0;
							}
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
												&user_event_queue)?;
						pending_count += 1;

						if pending_count >= save_batch_count {
							self.save(&mut evalutor,&checkpoint_path,&current_filename,current_item)?;
							pending_count = 0;
						}

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
			teachers.shuffle(&mut rng);

			let mut batch = Vec::with_capacity(learn_batch_size);

			for sfen in teachers.into_iter() {
				batch.push(sfen);

				if batch.len() == learn_batch_size {
					learning_process(&mut evalutor,
											batch,
											&user_event_queue)?;
					pending_count += 1;

					batch = Vec::with_capacity(learn_batch_size);
					count += learn_batch_size;
					current_item += learn_batch_size;

					if current_filename != "" && pending_count >= save_batch_count {
						self.save(&mut evalutor,&checkpoint_path,&current_filename,current_item)?;
						pending_count = 0;
					}
				}


				if let Err(ref e) = system_event_dispatcher.dispatch_events(&(), &*system_event_queue) {
					let _ = on_error_handler.lock().map(|h| h.call(e));
				}

				if notify_quit.load(Ordering::Acquire) {
					break;
				}
			}

			let remaing = batch.len();

			if !notify_quit.load(Ordering::Acquire) && remaing > 0 {
				learning_process(&mut evalutor,
							   		batch,
							   		&user_event_queue)?;
				pending_count += 1;

				if pending_count >= save_batch_count {
					self.save(&mut evalutor,&checkpoint_path,&current_filename,current_item)?;
					pending_count = 0;
				}
				count += remaing;
			}
		}

		if pending_count >= save_batch_count {
			self.save(&mut evalutor,&checkpoint_path,&current_filename,current_item)?;
		}

		if notify_run_test_arc.load(Ordering::Acquire) {
			let mut testdata = Vec::new();

			let mut paths = fs::read_dir(Path::new(&kifudir)
				.join("tests"))?.into_iter()
				.collect::<Vec<Result<DirEntry,_>>>();

			paths.sort_by(|a,b| {
				match (a,b) {
					(Ok(a),Ok(b)) => {
						let a = a.file_name();
						let b = b.file_name();
						a.cmp(&b)
					},
					_ => {
						std::cmp::Ordering::Equal
					}
				}
			});

			'test_files: for path in paths {
				let path = path?.path();

				if !path.as_path().extension().map(|e| e == ext).unwrap_or(false) {
					continue;
				}

				print!("{}\n", path.display());

				for b in BufReader::new(File::open(path)?).bytes() {
					let b = b?;

					record.push(b);

					if record.len() == item_size {
						testdata.push(record);
						record = Vec::with_capacity(item_size);
					} else {
						continue;
					}

					if testdata.len() >= 10000 {
						break 'test_files;
					}
				}
			}

			testdata.shuffle(&mut rng);

			let mut successed = 0;
			let mut count = 0;

			for packed in testdata.into_iter().take(100) {
				let (s,score) = test_process(&mut evalutor,packed)?;

				let success = match s {
					GameEndState::Win => {
						score >= 0.5
					},
					_ => {
						score < 0.5
					}
				};

				if success {
					successed += 1;
					println!("勝率{}% 正解!",score * 100.);
				} else {
					println!("勝率{}% 不正解...",score * 100.);
				}

				count += 1;
			}

			println!("正解率 {}%",successed as f32 / count as f32 * 100.);
		}

		print!("{}局面を学習しました。\n",count);

		Ok(())
	}

	fn learning_from_yaneuraou_bin_batch(evalutor:&mut Trainer<NN>,
										 batch:Vec<Vec<u8>>,
										 user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
	) -> Result<(),ApplicationError> {
		match evalutor.learning_by_packed_sfens(
			batch,
			&*user_event_queue) {
			Err(e) => {
				return Err(ApplicationError::LearningError(format!(
					"An error occurred while learning the neural network. {}",e
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
								user_event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>
	) -> Result<(),ApplicationError> {
		match evalutor.learning_by_hcpe(
			batch,
			 &*user_event_queue) {
			Err(e) => {
				return Err(ApplicationError::LearningError(format!(
					"An error occurred while learning the neural network. {}",e
				)));
			},
			Ok((msa,moa,msb,mob)) => {
				println!("error_total: {}, {}, {}, {}",msa, moa, msb, mob);
				Ok(())
			}
		}
	}

	fn save(&self,evalutor: &mut Trainer<NN>,checkpoint_path:&PathBuf,current_filename:&str,current_item:usize)
		-> Result<(),ApplicationError> {
		evalutor.save()?;

		let tmp_path = format!("{}.tmp",&checkpoint_path.as_path().to_string_lossy());
		let tmp_path = Path::new(&tmp_path);

		let mut checkpoint_writer = CheckPointWriter::new(tmp_path,&checkpoint_path.as_path())?;

		checkpoint_writer.save(&CheckPoint {
			filename:current_filename.to_string(),
			item:current_item
		})?;

		Ok(())
	}
}