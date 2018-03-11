use std::collections::HashMap;

pub struct TwoKeyHashMap<T> where T: Clone {
	map:HashMap<u64,Vec<(u64,T)>>
}

impl<T> TwoKeyHashMap<T> where T: Clone {
	pub fn new() -> TwoKeyHashMap<T> {
		let map:HashMap<u64,Vec<(u64,T)>> = HashMap::new();

		TwoKeyHashMap {
			map:map
		}
	}

	pub fn get(&self,k:&u64,sk:&u64) -> Option<T> where T: Clone {
		match self.map.get(k) {
			Some(v) if v.len() == 1 => {
				Some(v[0].1.clone())
			},
			Some(v) if v.len() > 1 => {
				for e in v {
					if e.0 == *sk {
						return Some(e.1.clone());
					}
				}
				None
			},
			_ => None,
		}
	}

	pub fn insert(&mut self,k:u64,sk:u64,nv:T) -> Option<T> {
		match self.map.get_mut(&k) {
			Some(ref mut v) if v.len() == 1 => {
				let old = v[0].1.clone();
				v[0] = (sk,nv);
				Some(old)
			},
			Some(ref mut v) if v.len() > 1 => {
				for i in 0..v.len() {
					if v[i].0 == sk {
						let old = v[i].1.clone();
						v[i] = (sk,nv);
						return Some(old);
					}
				}
				None
			},
			_ => None,
		}
	}

	pub fn clear(&mut self) {
		self.map.clear();
	}
}
impl<T> Clone for TwoKeyHashMap<T> where T: Clone {
	fn clone(&self) -> TwoKeyHashMap<T> {
		TwoKeyHashMap {
			map:self.map.clone()
		}
	}
}