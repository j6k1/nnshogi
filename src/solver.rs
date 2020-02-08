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

pub enum MaybeMate {
	Nomate,
	MateMoves(u32,Vec<LegalMove>),
	MaxDepth,
	MaxNodes,
	Timeout,
	Continuation,
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
							strict_moves:bool,
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
		match mate_strategy.preprocess(self,
								strict_moves,
								already_oute_kyokumen_map,
								hasher,0,
								check_timelimit,stop,
								event_queue,event_dispatcher) {
			MaybeMate::Continuation => (),
			r => {
				return r;
			}
		}

		match nomate_strategy.preprocess(self,
								strict_moves,
								already_oute_kyokumen_map,
								hasher,0,
								check_timelimit,stop,
								event_queue,event_dispatcher) {
			MaybeMate::Continuation => (),
			r => {
				return r;
			}
		}

		loop {
			match mate_strategy.resume(self,
										strict_moves,
										max_depth,
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

			match nomate_strategy.resume(self,
										strict_moves,
										max_depth,
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
				has_unknown:false
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
				has_unknown:false
			},  AscComparator,
				DescComparator)
		}

		#[inline]
		fn pop_stack(&mut self) {
			self.current_frame = self.stack.pop().expect("current stack is empty.");
		}

		pub fn resume<L,F,S>(&mut self,
							solver:&mut Solver<E>,
							strict_moves:bool,
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
			self.exec(solver,
						strict_moves,
						self.stack.len(),
						max_depth,
						max_nodes,
						already_oute_kyokumen_map,
						hasher,
						check_timelimit,
						stop,
						on_searchstart,
						event_queue,
						event_dispatcher)
		}

		fn exec<L,F,S>(&mut self,
							solver:&mut Solver<E>,
							strict_moves:bool,
							depth:usize,
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
		) -> MaybeMate where E: PlayerError,
								L: Logger,
								F: FnMut() -> bool,
								S: FnMut(u32,u64) {

			if depth > 0 {
				let mut has_unknown = false;

				let mut r = match self.exec(solver,
						strict_moves,
						depth-1,
						max_depth,
						max_nodes,
						already_oute_kyokumen_map,
						hasher,
						check_timelimit,
						stop,
						on_searchstart,
						event_queue,
						event_dispatcher) {
					r @ MaybeMate::Continuation if self.current_frame.mvs.len() > 0 => {
						return r
					},
					MaybeMate::MateMoves(d,mvs)=> {
						if self.stack.len() % 2 == 0 || (!self.current_frame.has_unknown && self.current_frame.mvs.len() == 0) {
							let mut mvs = mvs;
							self.current_frame.m.map(|m| mvs.insert(0,m));
							MaybeMate::MateMoves(d,mvs)
						} else if self.current_frame.mvs.len() > 0 {
							return MaybeMate::Continuation;
						} else {
							MaybeMate::Continuation
						}
					},
					r @ MaybeMate::Nomate => {
						if self.stack.len() % 2 != 0 || (!self.current_frame.has_unknown && self.current_frame.mvs.len() == 0) {
							r
						} else if self.current_frame.mvs.len() > 0 {
							return MaybeMate::Continuation;
						} else {
							MaybeMate::Continuation
						}
					},
					r => {
						r
					}
				};

				if let MaybeMate::Continuation = r {
					has_unknown = self.current_frame.has_unknown;
				}

				self.pop_stack();

				if let MaybeMate::MaxDepth = r {
					if self.current_frame.mvs.len() == 0 {
						has_unknown = true;
					} else {
						r = MaybeMate::Continuation;
					}
				}

				self.current_frame.has_unknown = has_unknown;

				r
			} else {
				let current_depth = self.stack.len() as u32;

				let r = if current_depth % 2 == 0 {
					self.oute_only(solver,
										strict_moves,
										max_depth, max_nodes,
										already_oute_kyokumen_map,
										hasher, current_depth as u32,
										check_timelimit, stop,
										on_searchstart,
										event_queue, event_dispatcher)
				} else {
					self.response_oute(solver,
										strict_moves,
										max_depth, max_nodes,
										already_oute_kyokumen_map,
										hasher, current_depth as u32,
										check_timelimit, stop,
										on_searchstart,
										event_queue, event_dispatcher)
				};

				if let MaybeMate::Continuation = r {
					if self.stack.len() == 0 && self.current_frame.mvs.len() == 0 {
						return MaybeMate::Nomate
					}
				}

				r
			}
		}

		pub fn preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								strict_moves:bool,
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
				self.oute_only_preprocess(solver,
											strict_moves,
											already_oute_kyokumen_map,
											hasher,current_depth,check_timelimit,stop,
											event_queue,event_dispatcher)
			}
		}

		fn response_oute_preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								_:bool,
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
								strict_moves:bool,
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
				already_oute_kyokumen_map.as_mut().map(|m| m.insert(teban,mhash,shash,true));
				return MaybeMate::MateMoves(current_depth,vec![]);
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

				let next = Rule::apply_move_none_check(&self.current_frame.state,teban,&self.current_frame.mc,m.to_applied_move());

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
										continue;
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

						let prev_frame = mem::replace(&mut self.current_frame, CheckmateStackFrame {
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
							has_unknown:false
						});

						match self.oute_only_preprocess(solver,
														strict_moves,
														already_oute_kyokumen_map,
														hasher,
														current_depth+1,
														check_timelimit,
														stop,
														event_queue,
														event_dispatcher) {
							r @ MaybeMate::Continuation => {
								self.stack.push(prev_frame);
								r
							},
							r => {
								self.current_frame = prev_frame;
								r
							}
						}
					}
				}
			}
		}

		fn oute_only_preprocess<L,F>(&mut self,
								solver:&mut Solver<E>,
								strict_moves:bool,
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
							return MaybeMate::MateMoves(current_depth,vec![]);
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
					if strict_moves {
						return MaybeMate::MateMoves(current_depth,vec![m]);
					}
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
							already_oute_kyokumen_map.as_mut().map(|m| m.insert(teban,mhash,shash,true));
							return MaybeMate::MateMoves(current_depth,vec![m]);
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
								strict_moves:bool,
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
				already_oute_kyokumen_map.as_mut().map(|m| m.insert(teban,mhash,shash,false));
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

						let prev_frame = mem::replace(&mut self.current_frame, CheckmateStackFrame {
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
							has_unknown:false
						});

						match self.response_oute_preprocess(solver,
														strict_moves,
														already_oute_kyokumen_map,
														hasher,
														current_depth+1,
														check_timelimit,
														stop,
														event_queue,
														event_dispatcher) {
							r @ MaybeMate::Continuation => {
								self.stack.push(prev_frame);
								r
							},
							r => {
								self.current_frame = prev_frame;
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
		has_unknown:bool
	}
}