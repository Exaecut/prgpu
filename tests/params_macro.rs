use prgpu::effect::{Ctx, Geometry, Host, HostCapabilities, Timing};
use prgpu::{BlendMode, Color, ParamValue, ParamsSpec, Point2, PopupOptions, Snapshot};

prgpu::params! {
	pub enum P {
		#[slider(label = "Strength", range = 0.0..=100.0, default = 50.0, percent, precision = 3)]
		Strength,
		#[checkbox(label = "Flag", default = true)]
		Flag,
		#[color(label = "Tint", default = "#FF8040")]
		Tint,
		#[angle(label = "Angle", default = 0.0)]
		Angle,
		#[point(label = "Anchor", default = (0.5, 0.5))]
		Anchor,
		#[popup(label = "Mode", options = ["A", "B", "C"], default = 1)]
		Mode,
		#[popup(label = "Quality", options = Quality, default = Quality::High)]
		Qual,
		#[blend_mode(label = "Blend", default = Multiply)]
		Blend,

		#[group("Group")]
		#[checkbox(label = "Grouped", default = false)]
		Grouped,
		#[group(end)]

		#[button(label = "Button")]
		Btn,
		#[button(label = "Process", disabled = is_busy())]
		ProcessBtn,
	}
}

fn is_busy() -> bool { false }

#[derive(Clone, Copy, PartialEq, Eq, Debug, prgpu::Popup)]
#[repr(u32)]
pub enum Quality {
	#[option("Draft")]
	Draft = 0,
	#[option("Balanced")]
	Balanced = 1,
	#[option("High")]
	High = 2,
}

// Discriminants start at 1 (0 is the AE input layer); markers carry the variant's ID.
#[test]
fn discriminants_and_markers() {
	assert_eq!(P::Strength as usize, 1);
	assert_eq!(usize::from(P::Flag), 2);
	assert_eq!(<Strength as prgpu::Param>::ID, P::Strength);
	assert_eq!(<Btn as prgpu::Param>::ID, P::Btn);
	// 11 leaf params + 2 synthesized group markers.
	assert_eq!(<P as ParamsSpec>::COUNT, 13);
}

fn snapshot() -> <P as ParamsSpec>::Snapshot {
	let mut s = <P as ParamsSpec>::Snapshot::default();
	s.set(P::Strength, ParamValue::Float(50.0));
	s.set(P::Flag, ParamValue::Bool(true));
	s.set(P::Tint, ParamValue::Color(Color::from_u8(255, 128, 64, 255)));
	s.set(P::Angle, ParamValue::Float(90.0));
	s.set(P::Anchor, ParamValue::Point(Point2::new(0.25, 0.75)));
	s.set(P::Mode, ParamValue::Index(2));
	s.set(P::Qual, ParamValue::Index(2));
	s.set(P::Blend, ParamValue::Index(BlendMode::Multiply as u32));
	s
}

#[test]
fn snapshot_round_trip_typing() {
	let s = snapshot();
	let caps = HostCapabilities::new(Host::AfterEffects, prgpu::Backend::Cpu);
	let ctx = Ctx::new(&s, Geometry::default(), Timing::default(), caps, false);

	assert_eq!(ctx.get(Strength), 50.0_f32);
	assert!(ctx.get(Flag));
	assert_eq!(ctx.get(Tint), Color::from_u8(255, 128, 64, 255));
	assert_eq!(ctx.get(Angle), 90.0_f32);
	assert_eq!(ctx.get(Anchor), Point2::new(0.25, 0.75));
	assert_eq!(ctx.get(Mode), 2_u32);
	assert_eq!(ctx.get(Qual), Quality::High);
	assert_eq!(ctx.get(Blend), BlendMode::Multiply);
}

#[test]
fn popup_derive_labels_and_clamp() {
	assert_eq!(<Quality as PopupOptions>::LABELS, &["Draft", "Balanced", "High"]);
	assert_eq!(Quality::from_index(2), Quality::High);
	assert_eq!(Quality::High.to_index(), 2);
	// Out-of-range clamps to the first variant.
	assert_eq!(Quality::from_index(99), Quality::Draft);
}

// Unset slots and missing-variant coercions fall back to Default.
#[test]
fn unset_slots_default() {
	let s = <P as ParamsSpec>::Snapshot::default();
	let caps = HostCapabilities::new(Host::AfterEffects, prgpu::Backend::Cpu);
	let ctx = Ctx::new(&s, Geometry::default(), Timing::default(), caps, false);
	assert_eq!(ctx.get(Strength), 0.0_f32);
	assert!(!ctx.get(Flag));
	assert_eq!(ctx.get(Tint), Color::default());
	assert_eq!(ctx.get(Qual), Quality::Draft);
}
