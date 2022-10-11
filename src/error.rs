use std::fmt;
use std::error;
use std::io;
use std::convert::From;
use std::num::ParseIntError;
use std::num::ParseFloatError;
use usiagent::event::{EventQueue, SystemEvent, SystemEventKind};
use usiagent::event::UserEventKind;
use usiagent::error::{EventDispatchError, USIAgentRunningError};
use usiagent::error::USIAgentStartupError;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use usiagent::error::SelfMatchRunningError;
use usiagent::error::TypeConvertError;
use usiagent::error::SfenStringConvertError;
use usiagent::error::KifuWriteError;
use csaparser::error::CsaParserError;
use nncombinator::error::{ConfigReadError, CudaError, DeviceError, EvaluateError, IndexOutBoundError, PersistenceError, TrainingError};

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

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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
impl From<TypeConvertError<String>> for CommonError {
	fn from(err: TypeConvertError<String>) -> CommonError {
		CommonError::Fail(format!("An error occurred during type conversion. ({})",err))
	}
}
impl From<io::Error> for CommonError {
	fn from(_: io::Error) -> CommonError {
		CommonError::Fail(String::from("I/O Error."))
	}
}
impl From<UsiProtocolError> for CommonError {
	fn from(err: UsiProtocolError) -> CommonError {
		match err {
			UsiProtocolError::InvalidState(s) => CommonError::Fail(s)
		}
	}
}
impl From<IndexOutBoundError> for CommonError {
	fn from(err: IndexOutBoundError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
impl From<TrainingError> for CommonError {
	fn from(err: TrainingError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
impl From<EvaluateError> for CommonError {
	fn from(err: EvaluateError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
impl From<ConfigReadError> for CommonError {
	fn from(err: ConfigReadError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
impl From<PersistenceError> for CommonError {
	fn from(err: PersistenceError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
impl<'a> From<ApplicationError> for CommonError {
	fn from(err: ApplicationError) -> Self {
		CommonError::Fail(format!("{}",err))
	}
}
#[derive(Debug)]
pub enum ApplicationError {
	StartupError(String),
	SfenStringConvertError(SfenStringConvertError),
	EventDispatchError(String),
	IOError(io::Error),
	ParseIntError(ParseIntError),
	ParseFloatError(ParseFloatError),
	AgentRunningError(String),
	SelfMatchRunningError(SelfMatchRunningError<CommonError>),
	CsaParserError(CsaParserError),
	LogicError(String),
	LearningError(String),
	KifuWriteError(KifuWriteError),
	SerdeError(toml::ser::Error),
	ConfigReadError(ConfigReadError),
	TrainingError(TrainingError),
	EvaluateError(EvaluateError),
	DeviceError(DeviceError),
	PersistenceError(PersistenceError),
	CudaError(CudaError),
}
impl fmt::Display for ApplicationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ApplicationError::StartupError(ref s) => write!(f, "{}",s),
			ApplicationError::SfenStringConvertError(ref e) => write!(f, "{}",e),
			ApplicationError::EventDispatchError(ref s) => write!(f,"{}",s),
			ApplicationError::IOError(ref e) => write!(f, "{}",e),
			ApplicationError::ParseIntError(ref e) => write!(f, "{}",e),
			ApplicationError::ParseFloatError(ref e) => write!(f, "{}",e),
			ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
			ApplicationError::SelfMatchRunningError(ref e) => write!(f, "{}",e),
			ApplicationError::CsaParserError(ref e) => write!(f, "{}",e),
			ApplicationError::LogicError(ref s) => write!(f,"{}",s),
			ApplicationError::LearningError(ref s) => write!(f,"{}",s),
			ApplicationError::KifuWriteError(ref s) => write!(f,"{}",s),
			ApplicationError::SerdeError(ref e) => write!(f,"{}",e),
			ApplicationError::ConfigReadError(ref e) => write!(f,"{}",e),
			ApplicationError::TrainingError(ref e) => write!(f,"{}",e),
			ApplicationError::EvaluateError(ref e) => write!(f,"{}",e),
			ApplicationError::DeviceError(ref e) => write!(f,"{}",e),
			ApplicationError::PersistenceError(ref e) => write!(f,"{}",e),
			ApplicationError::CudaError(ref e) => write!(f, "An error occurred in the process of cuda. ({})",e),
		}
	}
}
impl error::Error for ApplicationError {
	fn description(&self) -> &str {
		match *self {
			ApplicationError::StartupError(_) => "Startup Error.",
			ApplicationError::SfenStringConvertError(_) => "An error occurred during conversion to sfen string.",
			ApplicationError::EventDispatchError(_) => "An error occurred while processing the event.",
			ApplicationError::IOError(_) => "IO Error.",
			ApplicationError::ParseIntError(_) => "An error occurred parsing the integer string.",
			ApplicationError::ParseFloatError(_) => "An error occurred parsing the float string.",
			ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
			ApplicationError::SelfMatchRunningError(_) => "An error occurred while running the self-match.",
			ApplicationError::CsaParserError(_) => "An error occurred parsing the csa file.",
			ApplicationError::LogicError(_) => "Logic error.",
			ApplicationError::LearningError(_) => "An error occurred while learning the neural network.",
			ApplicationError::KifuWriteError(_) => "An error occurred when recording kifu or initialize KifuWriter.",
			ApplicationError::SerdeError(_) => "An error occurred during serialization or deserialization.",
			ApplicationError::ConfigReadError(_) => "An error occurred while loading the neural network model.",
			ApplicationError::TrainingError(_) => "An error occurred while training the model.",
			ApplicationError::EvaluateError(_) => "An error occurred when running the neural network.",
			ApplicationError::DeviceError(_) => "An error occurred during device initialization.",
			ApplicationError::PersistenceError(_) => "An error occurred when saving model information.",
			ApplicationError::CudaError(_) => "An error occurred in the process of cuda.",
		}
	}

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match *self {
			ApplicationError::StartupError(_) => None,
			ApplicationError::SfenStringConvertError(ref e) => Some(e),
			ApplicationError::EventDispatchError(_) => None,
			ApplicationError::IOError(ref e) => Some(e),
			ApplicationError::ParseIntError(ref e) => Some(e),
			ApplicationError::ParseFloatError(ref e) => Some(e),
			ApplicationError::AgentRunningError(_) => None,
			ApplicationError::SelfMatchRunningError(ref e) => Some(e),
			ApplicationError::CsaParserError(ref e) => Some(e),
			ApplicationError::LogicError(_) => None,
			ApplicationError::LearningError(_) => None,
			ApplicationError::KifuWriteError(ref e) => Some(e),
			ApplicationError::SerdeError(ref e) => Some(e),
			ApplicationError::ConfigReadError(ref e) => Some(e),
			ApplicationError::TrainingError(ref e) => Some(e),
			ApplicationError::EvaluateError(ref e) => Some(e),
			ApplicationError::DeviceError(ref e) => Some(e),
			ApplicationError::PersistenceError(ref e) => Some(e),
			ApplicationError::CudaError(_) => None
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
impl From<ParseFloatError> for ApplicationError {
	fn from(err: ParseFloatError) -> ApplicationError {
		ApplicationError::ParseFloatError(err)
	}
}
impl From<SelfMatchRunningError<CommonError>> for ApplicationError {
	fn from(err: SelfMatchRunningError<CommonError>) -> ApplicationError {
		ApplicationError::SelfMatchRunningError(err)
	}
}
impl From<SfenStringConvertError> for ApplicationError {
	fn from(err: SfenStringConvertError) -> ApplicationError {
		ApplicationError::SfenStringConvertError(err)
	}
}
impl From<KifuWriteError> for ApplicationError {
	fn from(err: KifuWriteError) -> ApplicationError {
		ApplicationError::KifuWriteError(err)
	}
}
impl<'a> From<EventDispatchError<'a,EventQueue<SystemEvent,SystemEventKind>,SystemEvent,CommonError>>
	for ApplicationError {
	fn from(err: EventDispatchError<'a, EventQueue<SystemEvent, SystemEventKind>, SystemEvent, CommonError>)
		-> ApplicationError {
		ApplicationError::EventDispatchError(format!("{}",err))
	}
}
impl From<CsaParserError> for ApplicationError {
	fn from(err: CsaParserError) -> ApplicationError {
		ApplicationError::CsaParserError(err)
	}
}
impl From<toml::ser::Error> for ApplicationError {
	fn from(err: toml::ser::Error) -> ApplicationError {
		ApplicationError::SerdeError(err)
	}
}
impl From<ConfigReadError> for ApplicationError {
	fn from(err: ConfigReadError) -> ApplicationError {
		ApplicationError::ConfigReadError(err)
	}
}
impl From<TrainingError> for ApplicationError {
	fn from(err: TrainingError) -> ApplicationError {
		ApplicationError::TrainingError(err)
	}
}
impl From<EvaluateError> for ApplicationError {
	fn from(err: EvaluateError) -> ApplicationError {
		ApplicationError::EvaluateError(err)
	}
}
impl From<DeviceError> for ApplicationError {
	fn from(err: DeviceError) -> ApplicationError {
		ApplicationError::DeviceError(err)
	}
}
impl From<PersistenceError> for ApplicationError {
	fn from(err: PersistenceError) -> ApplicationError {
		ApplicationError::PersistenceError(err)
	}
}
impl From<CudaError> for ApplicationError {
	fn from(err: CudaError) -> ApplicationError {
		ApplicationError::CudaError(err)
	}
}
