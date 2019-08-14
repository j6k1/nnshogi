use std::error;
use std::fmt;
use std::mem;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;

use usiagent::shogi::*;
use usiagent::rule::*;
use usiagent::hash::*;
use usiagent::logger::*;
use usiagent::event::*;
use usiagent::error::PlayerError;
use player::Search;

#[derive(Debug)]
pub enum SolveError {
	Timeout(String),
}
impl fmt::Display for SolveError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			SolveError::Timeout(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for SolveError {
	fn description(&self) -> &str {
		match *self {
			SolveError::Timeout(_) => "Time limit reached while checking if checkmate",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			SolveError::Timeout(_) => None,
		}
	}
}
pub enum MaybeMate {
	Mate(u32),
	Nomate,
	MateMoves(u32,Vec<LegalMove>),
	MaxDepth,
	MaxNodes,
	Timeout,
	Continuation,
	Unknown,
}
pub struct Solver<E> where E: PlayerError {
	error_type:PhantomData<E>
}
impl<E> Solver<E> where E: PlayerError {
	pub fn new()
	-> Solver<E> {
		Solver {
			error_type:PhantomData::<E>
		}
	}
	pub fn checkmate<L,F,S>(&mut self,
							teban:Teban,state:&State,
							mc:&MochigomaCollections,
							max_depth:Option<u32>,
							max_nodes:Option<u64>,
							oute_kyokumen_map:&mut KyokumenMap<u64,()>,
							already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
							current_kyokumen_map:&mut KyokumenMap<u64,u32>,
							hasher:&Search,
							mhash:u64,shash:u64,
							check_timelimit:&mut F,
							stop:&Arc<AtomicBool>,
							on_searchstart:&mut S,
							event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
							event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																UserEvent,Solver<E>,L,E>,)
	-> MaybeMate where E: PlayerError, L: Logger, F: FnMut() -> bool, S: FnMut(u32,u64) {
		let mvs = Rule::oute_only_moves_all(teban, state, mc);

		let mut mate_strategy = checkmate::MateStrategy::of_mate(teban,state.clone(),
															mc.clone(),mvs.clone(),
															mhash,shash,
															KyokumenMap::new(),
															oute_kyokumen_map.clone(),
															current_kyokumen_map.clone());


		let mut nomate_strategy = checkmate::NomateStrategy::of_nomate(teban,state.clone(),
															mc.clone(),mvs.clone(),
															mhash,shash,
															KyokumenMap::new(),
															oute_kyokumen_map.clone(),
															current_kyokumen_map.clone());
		if let MaybeMate::Nomate = mate_strategy.preprocess(self,already_oute_kyokumen_map,
								hasher,0,
								check_timelimit,stop,
								event_queue,event_dispatcher) {
			return MaybeMate::Nomate;
		}

		if let MaybeMate::Nomate = nomate_strategy.preprocess(self,already_oute_kyokumen_map,
								hasher,0,
								check_timelimit,stop,
								event_queue,event_dispatcher) {
			return MaybeMate::Nomate;
		}

		loop {
			match mate_strategy.exec(self,max_depth,
										max_nodes,
										already_oute_kyokumen_map,
										hasher,
										check_timelimit,
										stop,
										on_searchstart,
										event_queue,
										event_dispatcher) {
				MaybeMate::Continuation => (),
				r => {
					return r;
				}
			}

			match nomate_strategy.exec(self,max_depth,
										max_nodes,
										already_oute_kyokumen_map,
										hasher,
										check_timelimit,
										stop,
										on_searchstart,
										event_queue,
										event_dispatcher) {
				MaybeMate::Continuation => (),
				r => {
					return r;
				}
			}
		}
	}
}
pub trait Comparator<T>: Clone {
	fn cmp(&mut self,&T,&T) -> Ordering;
}
mod checkmate {
	use usiagent::TryFrom;
	use super::*;

	#[derive(Clone)]
	pub struct AscComparator;

	impl Comparator<(LegalMove,usize)> for AscComparator {
		#[inline]
		fn cmp(&mut self,l:&(LegalMove,usize),r:&(LegalMove,usize)) -> Ordering {
			l.1.cmp(&r.1)
		}
	}

	#[derive(Clone)]
	pub struct DescComparator;

	impl Comparator<(LegalMove,usize)> for DescComparator {
		#[inline]
		fn cmp(&mut self,l:&(LegalMove,usize),r:&(LegalMove,usize)) -> Ordering {
			r.1.cmp(&l.1)
		}
	}

	pub struct CheckmateStrategy<E,O,R>
			where E: PlayerError,
					O: Comparator<(LegalMove,usize)>,
					R: Comparator<(LegalMove,usize)> {
		error_type:PhantomData<E>,
		oute_comparator:O,
		response_oute_comparator:R,
		nodes:u64,
		current_frame:CheckmateStackFrame,
		stack:Vec<CheckmateStackFrame>,
	}

	pub type MateStrategy<E> = CheckmateStrategy<E,DescComparator,AscComparator>;
	pub type NomateStrategy<E> = CheckmateStrategy<E,AscComparator,DescComparator>;

	impl<E,O,R> CheckmateStrategy<E,O,R>
			where E: PlayerError,
					O: Comparator<(LegalMove,usize)>,
					R: Comparator<(LegalMove,usize)> {
		fn new(current_frame:CheckmateStackFrame,
				oute_comparator:O,
				response_oute_comparator:R,
		) -> CheckmateStrategy<E,O,R> {
			CheckmateStrategy {
				error_type:PhantomData::<E>,
				oute_comparator:oute_comparator,
				response_oute_comparator:response_oute_comparator,
				nodes:0,
				stack:Vec::new(),
				current_frame:current_frame,
			}
		}

		pub fn of_mate(teban:Teban,state:State,mc:MochigomaCollections,
						mvs:Vec<LegalMove>,mhash:u64,shash:u64,
						ignore_kyokumen_map:KyokumenMap<u64,()>,
						oute_kyokumen_map:KyokumenMap<u64,()>,
						current_kyokumen_map:KyokumenMap<u64,u32>)
		-> CheckmateStrategy<E,DescComparator,AscComparator> {
			CheckmateStrategy::new(CheckmateStackFrame {
				teban:teban,
				state:state,
				mc:mc,
				mvs:mvs,
				m:None,
				mhash:mhash,
				shash:shash,
				ignore_kyokumen_map:ignore_kyokumen_map,
				oute_kyokumen_map:oute_kyokumen_map,
				current_kyokumen_map:current_kyokumen_map,
			},  DescComparator,
				AscComparator)
		}

		pub fn of_nomate(teban:Teban,state:State,mc:MochigomaCollections,
						mvs:Vec<LegalMove>,mhash:u64,shash:u64,
						ignore_kyokumen_map:KyokumenMap<u64,()>,
						oute_kyokumen_map:KyokumenMap<u64,()>,
						current_kyokumen_map:KyokumenMap<u64,u32>)
		-> CheckmateStrategy<E,AscComparator,DescComparator> {
			CheckmateStrategy::new(CheckmateStackFrame {
				teban:teban,
				state:state,
				mc:mc,
				mvs:mvs,
				m:None,
				mhash:mhash,
				shash:shash,
				ignore_kyokumen_map:ignore_kyokumen_map,
				oute_kyokumen_map:oute_kyokumen_map,
				current_kyokumen_map:current_kyokumen_map,
			},  AscComparator,
				DescComparator)
		}

		#[inline]
		fn pop_stack(&mut self) {
			self.current_frame = self.stack.pop().expect("current stack is empty.");
		}

		pub fn exec<L,F,S>(&mut self,
							solver:&mut Solver<E>,
							max_depth:Option<u32>,
							max_nodes:Option<u64>,
							already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
							hasher:&Search,
							check_timelimit:&mut F,
							stop:&Arc<AtomicBool>,
							on_searchstart:&mut S,
							event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
							event_dispatcher:&mut USIEventDispatcher<UserEventKind,
											UserEvent,Solver<E>,L,E>
		)  -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool,
								S: FnMut(u32,u64) {
			let current_depth = self.stack.len() as u32;

			if current_depth % 2 == 0 {
				if self.stack.len() == 0 && self.current_frame.mvs.len() ==0 {
					return MaybeMate::Unknown;
				}

				let r = self.oute_only(solver,max_depth, max_nodes,
											already_oute_kyokumen_map,
											hasher, current_depth as u32 + 1,
											check_timelimit, stop,
											on_searchstart,
											event_queue, event_dispatcher);
				match r {
					MaybeMate::Mate(depth) => {
						let depth = depth - 1;

						already_oute_kyokumen_map.as_mut().map(|m| {
							m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
						});

						let mut mvs = Vec::new();

						let m = if depth > current_depth && current_depth >= 2 {
							let mut depth = depth;

							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});

							if depth % 2 == 0 {
								self.pop_stack();
								depth -= 1;
							}

							while depth > current_depth {
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
								});

								mvs.insert(0, self.current_frame.m.expect("current move is none."));
								self.pop_stack();
								depth -= 1;
							}

							let m = self.current_frame.m;

							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});

							mvs.insert(0, self.current_frame.m.expect("current move is none."));
							self.pop_stack();

							m
						} else if current_depth == 0 {
							return MaybeMate::MateMoves(depth,vec![]);
						} else {
							let m = self.current_frame.m;

							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});

							self.pop_stack();
							m
						};

						let is_put_fu = match m {
							Some(LegalMove::Put(ref m)) if m.kind() == MochigomaKind::Fu => true,
							_ => false,
						};

						if self.stack.len() == 0 {
							MaybeMate::MateMoves(depth,mvs)
						} else if !is_put_fu {
							mvs.insert(0, self.current_frame.m.expect("current move is none."));

							while self.current_frame.mvs.len() == 0 {
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
								});
								if self.stack.len() < 2 {
									self.pop_stack();
									already_oute_kyokumen_map.as_mut().map(|m| {
										m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
									});
									return MaybeMate::MateMoves(depth,mvs);
								}
								self.pop_stack();
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
								});
								mvs.insert(0, self.current_frame.m.expect("current move is none."));
								self.pop_stack();
								mvs.insert(0, self.current_frame.m.expect("current move is none."));
							}
							MaybeMate::Continuation
						} else {
							MaybeMate::Continuation
						}
					},
					MaybeMate::Nomate => {
						if self.stack.len() == 0 {
							if self.current_frame.mvs.len() == 0 {
								MaybeMate::Nomate
							} else {
								MaybeMate::Continuation
							}
						} else {
							while self.current_frame.mvs.len() == 0 {
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,false)
								});
								if self.stack.len() < 2 {
									return MaybeMate::Nomate;
								} else {
									self.pop_stack();
									already_oute_kyokumen_map.as_mut().map(|m| {
										m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,false)
									});
									self.pop_stack();
								}
							}

							if self.stack.len() == 0 {
								MaybeMate::Nomate
							}  else {
								MaybeMate::Continuation
							}
						}
					},
					MaybeMate::MaxDepth => {
						if self.stack.len() > 0 {
							self.pop_stack();
						}

						while self.stack.len() > 0 && self.current_frame.mvs.len() == 0 {
							self.pop_stack();
						}

						MaybeMate::MaxDepth
					},
					r => {
						r
					}
				}
			} else {
				let r = self.response_oute(solver,max_depth, max_nodes,
											already_oute_kyokumen_map,
											hasher, current_depth as u32 + 1,
											check_timelimit, stop,
											on_searchstart,
											event_queue, event_dispatcher);
				match r {
					MaybeMate::Nomate => {
						already_oute_kyokumen_map.as_mut().map(|m| {
							m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,false)
						});
						self.pop_stack();

						while self.current_frame.mvs.len() == 0 {
							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,false)
							});
							if self.stack.len() < 2 {
								return MaybeMate::Nomate;
							}
							self.pop_stack();
							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,false)
							});
							self.pop_stack();
						}

						MaybeMate::Continuation
					},
					MaybeMate::Mate(depth) => {
						let depth = depth - 1;

						let mut mvs = Vec::new();

						if depth > current_depth {
							let mut depth = depth;

							if depth % 2 == 0 {
								self.pop_stack();
								depth -= 1;
							}

							while depth > current_depth {
								mvs.insert(0, self.current_frame.m.expect("current move is none."));
								self.pop_stack();
								depth -= 1;
							}
						}

						mvs.insert(0, self.current_frame.m.expect("current move is none."));

						if self.current_frame.mvs.len() == 0 {
							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});

							if self.stack.len() < 2 {
								self.pop_stack();
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
								});
								return MaybeMate::MateMoves(depth,mvs);
							}
							self.pop_stack();

							let m = self.current_frame.m;

							let is_put_fu = match m {
								Some(LegalMove::Put(ref m)) if m.kind() == MochigomaKind::Fu => true,
								_ => false,
							};

							if is_put_fu {
								return MaybeMate::Continuation;
							}

							if self.stack.len() == 0 {
								return MaybeMate::MateMoves(depth,mvs);
							}

							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});

							mvs.insert(0, self.current_frame.m.expect("current move is none."));
							self.pop_stack();
							mvs.insert(0, self.current_frame.m.expect("current move is none."));
						} else {
							return MaybeMate::Continuation;
						}

						while self.current_frame.mvs.len() == 0 {
							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});
							if self.stack.len() < 2 {
								self.pop_stack();
								already_oute_kyokumen_map.as_mut().map(|m| {
									m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
								});
								return MaybeMate::MateMoves(depth,mvs);
							}

							self.pop_stack();
							mvs.insert(0, self.current_frame.m.expect("current move is none."));
							already_oute_kyokumen_map.as_mut().map(|m| {
								m.insert(self.current_frame.teban,self.current_frame.mhash,self.current_frame.shash,true)
							});
							self.pop_stack();
							mvs.insert(0, self.current_frame.m.expect("current move is none."));
						}
						MaybeMate::Continuation
					},
					MaybeMate::MaxDepth => {
						if self.stack.len() > 0 {
							self.pop_stack();
						}

						while self.stack.len() > 0 && self.current_frame.mvs.len() == 0 {
							self.pop_stack();
						}

						MaybeMate::MaxDepth
					},
					r => {
						r
					}
				}
			}
		}

		pub fn preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
								hasher:&Search,
								current_depth:u32,
								check_timelimit:&mut F,
								stop:&Arc<AtomicBool>,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																	UserEvent,Solver<E>,L,E>
		)  -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool {
			if self.current_frame.mvs.len() == 0 {
				MaybeMate::Nomate
			} else {
				self.oute_only_preprocess(solver,already_oute_kyokumen_map,
											hasher,current_depth,check_timelimit,stop,
											event_queue,event_dispatcher)
			}
		}

		fn response_oute_preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
								hasher:&Search,
								_:u32,
								check_timelimit:&mut F,
								stop:&Arc<AtomicBool>,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																	UserEvent,Solver<E>,L,E>
		)  -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool {

			let teban = self.current_frame.teban;
			let state = &self.current_frame.state;
			let mc = &self.current_frame.mc;
			let mhash = self.current_frame.mhash;
			let shash = self.current_frame.shash;
			let ignore_kyokumen_map = &self.current_frame.ignore_kyokumen_map;
			let current_kyokumen_map = &self.current_frame.current_kyokumen_map;
			let mut pmvs = Vec::with_capacity(self.current_frame.mvs.len());
			let mvs = mem::replace(&mut self.current_frame.mvs, Vec::new());

			for m in mvs.into_iter() {
				let o = match m {
					LegalMove::To(ref m) => {
						match m.obtained() {
							Some(ObtainKind::Ou) => {
								return MaybeMate::Nomate;
							},
							Some(o) => {
								MochigomaKind::try_from(o).ok()
							},
							None => None,
						}
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
					m.get(teban,&mhash,&shash)
				});

				if let Some(true) = completed {
					continue;
				} else if let Some(false) = completed {
					return MaybeMate::Nomate;
				}

				if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				}

				if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
					if c >= 3 {
						continue;
					}
				}

				let next = Rule::apply_move_none_check(state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) => {
						let len = Rule::oute_only_moves_all(teban.opposite(), next, mc).len();

						if len == 0 {
							return MaybeMate::Nomate;
						}

						pmvs.push((m,len));
					}
				}

				let _ = event_dispatcher.dispatch_events(solver,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			let mut comparator = self.response_oute_comparator.clone();

			pmvs.sort_by(|a,b| comparator.cmp(a,b));

			self.current_frame.mvs = pmvs.into_iter().map(|(m,_)| m).collect::<Vec<LegalMove>>();

			MaybeMate::Continuation
		}

		fn response_oute<L,F,S>(&mut self,
								solver:&mut Solver<E>,
								max_depth:Option<u32>,
								max_nodes:Option<u64>,
								already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
								hasher:&Search,
								current_depth:u32,
								check_timelimit:&mut F,
								stop:&Arc<AtomicBool>,
								on_searchstart:&mut S,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																	UserEvent,Solver<E>,L,E>
		)  -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool,
								S: FnMut(u32,u64) {
			self.nodes += 1;

			let teban = self.current_frame.teban;
			let mhash = self.current_frame.mhash;
			let shash = self.current_frame.shash;
			let mut ignore_kyokumen_map = self.current_frame.ignore_kyokumen_map.clone();
			let mut oute_kyokumen_map = self.current_frame.oute_kyokumen_map.clone();
			let mut current_kyokumen_map = self.current_frame.current_kyokumen_map.clone();

			on_searchstart(current_depth,self.nodes);

			if max_depth.map(|d| current_depth >= d).unwrap_or(false) {
				return MaybeMate::MaxDepth;
			}

			if max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
				return MaybeMate::MaxNodes;
			}

			let _ = event_dispatcher.dispatch_events(solver,&*event_queue);

			if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
				return MaybeMate::Timeout;
			}

			if self.current_frame.mvs.len() == 0 {
				return MaybeMate::Mate(current_depth);
			} else {
				let m = self.current_frame.mvs.remove(0);

				let o = match m {
					LegalMove::To(ref m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,
													self.current_frame.state.get_banmen(),
													& self.current_frame.mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,
													self.current_frame.state.get_banmen(),
													& self.current_frame.mc,&m.to_move(),&o);

				ignore_kyokumen_map.insert(teban,mhash,shash,());

				match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
					&c => {
						current_kyokumen_map.insert(teban, mhash, shash, c+1);
					}
				}

				let next = Rule::apply_move_none_check(& self.current_frame.state,teban,& self.current_frame.mc,m.to_applied_move());

				match next {
					(next,nmc,_) => {
						{
							let (x,y,kind) = match m {
								LegalMove::To(ref mv) => {
									let (dx,dy) = mv.dst().square_to_point();
									let kind = next.get_banmen().0[dy as usize][dx as usize];

									(dx,dy,kind)
								},
								LegalMove::Put(ref mv) => {
									let kind = KomaKind::from((teban,mv.kind()));
									let (dx,dy) = mv.dst().square_to_point();

									(dx,dy,kind)
								}
							};

							let ps = next.get_part();

							if Rule::is_mate_with_partial_state_and_point_and_kind(teban,ps,x,y,kind) ||
							   Rule::is_mate_with_partial_state_repeat_move_kinds(teban,ps) {

								match oute_kyokumen_map.get(teban,&mhash,&shash) {
									Some(_) => {
										return MaybeMate::Nomate;
									},
									None => {
										oute_kyokumen_map.insert(teban,mhash,shash,());
									},
								}
							} else {
								oute_kyokumen_map.clear(teban);
							}
						}

						let mvs = Rule::oute_only_moves_all(teban.opposite(), &next, &nmc);

						self.stack.push(mem::replace(&mut self.current_frame, CheckmateStackFrame {
							teban:teban.opposite(),
							state:next,
							mc:nmc,
							mvs:mvs,
							m:Some(m),
							mhash:mhash,
							shash:shash,
							ignore_kyokumen_map:ignore_kyokumen_map,
							oute_kyokumen_map:oute_kyokumen_map,
							current_kyokumen_map:current_kyokumen_map,
						}));

						let r = self.oute_only_preprocess(solver,already_oute_kyokumen_map,
														hasher,
														current_depth+1,
														check_timelimit,
														stop,
														event_queue,
														event_dispatcher);
						match r {
							MaybeMate::Nomate => {
								self.pop_stack();
								r
							},
							r => {
								r
							}
						}
					}
				}
			}
		}

		fn oute_only_preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
								hasher:&Search,
								current_depth:u32,
								check_timelimit:&mut F,
								stop:&Arc<AtomicBool>,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																	UserEvent,Solver<E>,L,E>
		)  -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool {

			let teban = self.current_frame.teban;
			let state = &self.current_frame.state;
			let mc = &self.current_frame.mc;
			let mhash = self.current_frame.mhash;
			let shash = self.current_frame.shash;
			let ignore_kyokumen_map = &self.current_frame.ignore_kyokumen_map;
			let current_kyokumen_map = &self.current_frame.current_kyokumen_map;
			let mut pmvs = Vec::with_capacity(self.current_frame.mvs.len());
			let oute_kyokumen_map = &self.current_frame.oute_kyokumen_map;
			let mvs = mem::replace(&mut self.current_frame.mvs, Vec::new());

			for m in mvs.into_iter() {
				match m {
					LegalMove::To(ref m) => {
						if let Some(ObtainKind::Ou) = m.obtained() {
							already_oute_kyokumen_map.as_mut().map(|m| m.insert(teban,mhash,shash,true));
							return MaybeMate::Mate(current_depth);
						}
					},
					_ => ()
				}

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				let o = match m {
					LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				let completed = already_oute_kyokumen_map.as_ref().and_then(|m| {
					m.get(teban,&mhash,&shash)
				});

				if let Some(true) = completed {
					return MaybeMate::Mate(current_depth);
				} else if let Some(false) = completed {
					continue;
				}

				if let Some(()) = ignore_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				}

				if let Some(&c) = current_kyokumen_map.get(teban,&mhash,&shash) {
					if c >= 3 {
						continue;
					}
				}

				if let Some(()) = oute_kyokumen_map.get(teban,&mhash,&shash) {
					continue;
				}

				match next {
					(ref next,ref mc,_) => {
						let len = Rule::respond_oute_only_moves_all(teban.opposite(), next, mc).len();


						if len == 0 {
							return MaybeMate::Mate(current_depth);
						}

						pmvs.push((m,len));
					}
				}

				let _ = event_dispatcher.dispatch_events(solver,&*event_queue);

				if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
					return MaybeMate::Timeout;
				}
			}

			let mut comparator = self.oute_comparator.clone();

			pmvs.sort_by(|a,b| comparator.cmp(a,b));

			self.current_frame.mvs = pmvs.into_iter().map(|(m,_)| m).collect::<Vec<LegalMove>>();

			MaybeMate::Continuation
		}

		fn oute_only<L,F,S>(&mut self,
								solver:&mut Solver<E>,
								max_depth:Option<u32>,
								max_nodes:Option<u64>,
								already_oute_kyokumen_map:&mut Option<KyokumenMap<u64,bool>>,
								hasher:&Search,
								current_depth:u32,
								check_timelimit:&mut F,
								stop:&Arc<AtomicBool>,
								on_searchstart:&mut S,
								event_queue:&Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
								event_dispatcher:&mut USIEventDispatcher<UserEventKind,
																	UserEvent,Solver<E>,L,E>,
		) -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool,
								S: FnMut(u32,u64) {
			self.nodes += 1;

			let teban = self.current_frame.teban;
			let mhash = self.current_frame.mhash;
			let shash = self.current_frame.shash;
			let mut ignore_kyokumen_map = self.current_frame.ignore_kyokumen_map.clone();
			let mut oute_kyokumen_map = self.current_frame.oute_kyokumen_map.clone();
			let mut current_kyokumen_map = self.current_frame.current_kyokumen_map.clone();

			on_searchstart(current_depth,self.nodes);

			if max_depth.map(|d| current_depth >= d).unwrap_or(false) {
				return MaybeMate::MaxDepth;
			}

			if max_nodes.map(|n| self.nodes >= n).unwrap_or(false) {
				return MaybeMate::MaxNodes;
			}

			let _ = event_dispatcher.dispatch_events(solver,&*event_queue);

			if check_timelimit() || stop.load(atomic::Ordering::Acquire) {
				return MaybeMate::Timeout;
			}

			if self.current_frame.mvs.len() == 0 {
				return MaybeMate::Nomate;
			} else {
				let m = self.current_frame.mvs.remove(0);

				let next = Rule::apply_move_none_check(&self.current_frame.state,teban,&self.current_frame.mc,m.to_applied_move());

				let o = match m {
					LegalMove::To(ref m) => {
						m.obtained().and_then(|o| MochigomaKind::try_from(o).ok())
					},
					_ => None,
				};

				let mhash = hasher.calc_main_hash(mhash,&teban,
													self.current_frame.state.get_banmen(),
													&self.current_frame.mc,&m.to_move(),&o);
				let shash = hasher.calc_sub_hash(shash,&teban,
													self.current_frame.state.get_banmen(),
													&self.current_frame.mc,&m.to_move(),&o);

				ignore_kyokumen_map.insert(teban,mhash,shash,());

				match current_kyokumen_map.get(teban, &mhash, &shash).unwrap_or(&0) {
					&c => {
						current_kyokumen_map.insert(teban, mhash, shash, c+1);
					}
				}

				oute_kyokumen_map.insert(teban, mhash, shash, ());

				match next {
					(next, nmc,_) => {
						let mvs = Rule::respond_oute_only_moves_all(teban.opposite(), &next, &nmc);

						self.stack.push(mem::replace(&mut self.current_frame, CheckmateStackFrame {
							teban:teban.opposite(),
							state:next,
							mc:nmc,
							mvs:mvs,
							m:Some(m),
							mhash:mhash,
							shash:shash,
							ignore_kyokumen_map:ignore_kyokumen_map,
							oute_kyokumen_map:oute_kyokumen_map,
							current_kyokumen_map:current_kyokumen_map,
						}));

						let r = self.response_oute_preprocess(solver,already_oute_kyokumen_map,
														hasher,
														current_depth+1,
														check_timelimit,
														stop,
														event_queue,
														event_dispatcher);

						match r {
							MaybeMate::Nomate => {
								self.pop_stack();
								r
							},
							r => {
								r
							}
						}
					}
				}
			}
		}
	}
	#[derive(Clone)]
	struct CheckmateStackFrame {
		teban:Teban,state:State,
		mc:MochigomaCollections,
		mvs:Vec<LegalMove>,
		m:Option<LegalMove>,
		mhash:u64,shash:u64,
		ignore_kyokumen_map:KyokumenMap<u64,()>,
		oute_kyokumen_map:KyokumenMap<u64,()>,
		current_kyokumen_map:KyokumenMap<u64,u32>,
	}
}