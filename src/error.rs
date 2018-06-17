use std::fmt;
use std::error;
use std::io;
use std::convert::From;
use std::num::ParseIntError;
use usiagent::event::SystemEventKind;
use usiagent::event::UserEventKind;
use usiagent::error::USIAgentRunningError;
use usiagent::error::USIAgentStartupError;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use simplenn::error::InvalidStateError;
use simplenn::error::PersistenceError;

#[derive(Debug)]
pub enum CommonError {
	Fail(String),
}
impl PlayerError for CommonError {

}
impl fmt::Display for CommonError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			CommonError::Fail(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for CommonError {
	fn description(&self) -> &str {
		match *self {
			CommonError::Fail(_) => "Player error.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			CommonError::Fail(_) => None,
		}
	}
}
impl<'a> From<CommonError> for USIAgentRunningError<'a,SystemEventKind,CommonError>
	where SystemEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentRunningError<'a,SystemEventKind,CommonError> {
		USIAgentRunningError::from(USIAgentStartupError::PlayerError(err))
	}
}
impl<'a> From<CommonError> for USIAgentRunningError<'a,UserEventKind,CommonError>
	where UserEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentRunningError<'a,UserEventKind,CommonError> {
		USIAgentRunningError::from(USIAgentStartupError::PlayerError(err))
	}
}
impl From<InvalidStateError> for CommonError {
	fn from(_: InvalidStateError) -> CommonError {
		CommonError::Fail(String::from("invalid state."))
	}
}
impl From<io::Error> for CommonError {
	fn from(_: io::Error) -> CommonError {
		CommonError::Fail(String::from("I/O Error."))
	}
}
impl<E> From<PersistenceError<E>> for CommonError {
	fn from(_: PersistenceError<E>) -> CommonError {
		CommonError::Fail(String::from("An error occurred while saving model of NN."))
	}
}
impl From<UsiProtocolError> for CommonError {
	fn from(err: UsiProtocolError) -> CommonError {
		match err {
			UsiProtocolError::InvalidState(s) => CommonError::Fail(s)
		}
	}
}
#[derive(Debug)]
pub enum ApplicationError {
	StartupError(String),
	IOError(io::Error),
	ParseIntError(ParseIntError),
	AgentRunningError(String),
	SelfMatchRunningError(String),
}
impl fmt::Display for ApplicationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ApplicationError::StartupError(ref s) => write!(f, "{}",s),
			ApplicationError::IOError(ref e) => write!(f, "{}",e),
			ApplicationError::ParseIntError(ref e) => write!(f, "{}",e),
			ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
			ApplicationError::SelfMatchRunningError(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for ApplicationError {
	fn description(&self) -> &str {
		match *self {
			ApplicationError::StartupError(_) => "Startup Error.",
			ApplicationError::IOError(_) => "IO Error.",
			ApplicationError::ParseIntError(_) => "An error occurred parsing the integer string.",
			ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
			ApplicationError::SelfMatchRunningError(_) => "An error occurred while running the self-match.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			ApplicationError::StartupError(_) => None,
			ApplicationError::IOError(ref e) => Some(e),
			ApplicationError::ParseIntError(ref e) => Some(e),
			ApplicationError::AgentRunningError(_) => None,
			ApplicationError::SelfMatchRunningError(_) => None,
		}
	}
}
impl From<io::Error> for ApplicationError {
	fn from(err: io::Error) -> ApplicationError {
		ApplicationError::IOError(err)
	}
}
impl From<ParseIntError> for ApplicationError {
	fn from(err: ParseIntError) -> ApplicationError {
		ApplicationError::ParseIntError(err)
	}
}
