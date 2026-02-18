/**
 * Video Overlay Types
 *
 * Data structures for real-time video annotation overlays including
 * pose skeleton, head/gaze visualization, and event callouts.
 */

// ═══════════════════════════════════════════════════════════════
// POSE LANDMARKS (MediaPipe 33-point pose)
// ═══════════════════════════════════════════════════════════════

/**
 * Single pose landmark with position and visibility.
 * Coordinates are normalized 0-1 relative to video dimensions.
 */
export type PoseLandmark = [
  x: number,      // 0-1 normalized x position
  y: number,      // 0-1 normalized y position
  z: number,      // depth (relative to hip center)
  visibility: number  // 0-1 confidence
];

/**
 * MediaPipe pose landmark indices for reference.
 * Full 33-point pose model.
 */
export const POSE_LANDMARKS = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

/**
 * Skeleton connections for drawing limb lines.
 * Each pair represents [from_landmark, to_landmark].
 */
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Face
  [POSE_LANDMARKS.NOSE, POSE_LANDMARKS.LEFT_EYE_INNER],
  [POSE_LANDMARKS.LEFT_EYE_INNER, POSE_LANDMARKS.LEFT_EYE],
  [POSE_LANDMARKS.LEFT_EYE, POSE_LANDMARKS.LEFT_EYE_OUTER],
  [POSE_LANDMARKS.LEFT_EYE_OUTER, POSE_LANDMARKS.LEFT_EAR],
  [POSE_LANDMARKS.NOSE, POSE_LANDMARKS.RIGHT_EYE_INNER],
  [POSE_LANDMARKS.RIGHT_EYE_INNER, POSE_LANDMARKS.RIGHT_EYE],
  [POSE_LANDMARKS.RIGHT_EYE, POSE_LANDMARKS.RIGHT_EYE_OUTER],
  [POSE_LANDMARKS.RIGHT_EYE_OUTER, POSE_LANDMARKS.RIGHT_EAR],
  [POSE_LANDMARKS.MOUTH_LEFT, POSE_LANDMARKS.MOUTH_RIGHT],
  // Torso
  [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER],
  [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_HIP],
  [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_HIP],
  [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP],
  // Left arm
  [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW],
  [POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST],
  [POSE_LANDMARKS.LEFT_WRIST, POSE_LANDMARKS.LEFT_PINKY],
  [POSE_LANDMARKS.LEFT_WRIST, POSE_LANDMARKS.LEFT_INDEX],
  [POSE_LANDMARKS.LEFT_WRIST, POSE_LANDMARKS.LEFT_THUMB],
  [POSE_LANDMARKS.LEFT_PINKY, POSE_LANDMARKS.LEFT_INDEX],
  // Right arm
  [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW],
  [POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST],
  [POSE_LANDMARKS.RIGHT_WRIST, POSE_LANDMARKS.RIGHT_PINKY],
  [POSE_LANDMARKS.RIGHT_WRIST, POSE_LANDMARKS.RIGHT_INDEX],
  [POSE_LANDMARKS.RIGHT_WRIST, POSE_LANDMARKS.RIGHT_THUMB],
  [POSE_LANDMARKS.RIGHT_PINKY, POSE_LANDMARKS.RIGHT_INDEX],
  // Left leg
  [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE],
  [POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE],
  [POSE_LANDMARKS.LEFT_ANKLE, POSE_LANDMARKS.LEFT_HEEL],
  [POSE_LANDMARKS.LEFT_ANKLE, POSE_LANDMARKS.LEFT_FOOT_INDEX],
  [POSE_LANDMARKS.LEFT_HEEL, POSE_LANDMARKS.LEFT_FOOT_INDEX],
  // Right leg
  [POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE],
  [POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE],
  [POSE_LANDMARKS.RIGHT_ANKLE, POSE_LANDMARKS.RIGHT_HEEL],
  [POSE_LANDMARKS.RIGHT_ANKLE, POSE_LANDMARKS.RIGHT_FOOT_INDEX],
  [POSE_LANDMARKS.RIGHT_HEEL, POSE_LANDMARKS.RIGHT_FOOT_INDEX],
];

// ═══════════════════════════════════════════════════════════════
// HAND LANDMARKS (MediaPipe 21-point hand model)
// ═══════════════════════════════════════════════════════════════

/**
 * Single hand landmark with position and visibility.
 * Same format as pose landmarks.
 */
export type HandLandmark = [
  x: number,
  y: number,
  z: number,
  visibility: number
];

/**
 * MediaPipe hand landmark indices for reference.
 */
export const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

/**
 * Hand skeleton connections for drawing.
 */
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.THUMB_CMC],
  [HAND_LANDMARKS.THUMB_CMC, HAND_LANDMARKS.THUMB_MCP],
  [HAND_LANDMARKS.THUMB_MCP, HAND_LANDMARKS.THUMB_IP],
  [HAND_LANDMARKS.THUMB_IP, HAND_LANDMARKS.THUMB_TIP],
  // Index finger
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.INDEX_MCP],
  [HAND_LANDMARKS.INDEX_MCP, HAND_LANDMARKS.INDEX_PIP],
  [HAND_LANDMARKS.INDEX_PIP, HAND_LANDMARKS.INDEX_DIP],
  [HAND_LANDMARKS.INDEX_DIP, HAND_LANDMARKS.INDEX_TIP],
  // Middle finger
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.MIDDLE_MCP],
  [HAND_LANDMARKS.MIDDLE_MCP, HAND_LANDMARKS.MIDDLE_PIP],
  [HAND_LANDMARKS.MIDDLE_PIP, HAND_LANDMARKS.MIDDLE_DIP],
  [HAND_LANDMARKS.MIDDLE_DIP, HAND_LANDMARKS.MIDDLE_TIP],
  // Ring finger
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.RING_MCP],
  [HAND_LANDMARKS.RING_MCP, HAND_LANDMARKS.RING_PIP],
  [HAND_LANDMARKS.RING_PIP, HAND_LANDMARKS.RING_DIP],
  [HAND_LANDMARKS.RING_DIP, HAND_LANDMARKS.RING_TIP],
  // Pinky
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.PINKY_MCP],
  [HAND_LANDMARKS.PINKY_MCP, HAND_LANDMARKS.PINKY_PIP],
  [HAND_LANDMARKS.PINKY_PIP, HAND_LANDMARKS.PINKY_DIP],
  [HAND_LANDMARKS.PINKY_DIP, HAND_LANDMARKS.PINKY_TIP],
  // Palm
  [HAND_LANDMARKS.INDEX_MCP, HAND_LANDMARKS.MIDDLE_MCP],
  [HAND_LANDMARKS.MIDDLE_MCP, HAND_LANDMARKS.RING_MCP],
  [HAND_LANDMARKS.RING_MCP, HAND_LANDMARKS.PINKY_MCP],
];

// ═══════════════════════════════════════════════════════════════
// FRAME DATA
// ═══════════════════════════════════════════════════════════════

/**
 * Single frame of overlay data.
 */
export type OverlayFrame = {
  t_sec: number;                          // Timestamp in seconds
  pose: PoseLandmark[] | null;            // 33 pose landmarks or null if not detected
  child_bbox: number[] | null;            // [x, y, w, h, confidence] or null
  head_yaw: number | null;                // Head rotation in degrees (- = left, + = right)
  // Parent data
  parent_bbox: number[] | null;           // [x, y, w, h, confidence] or null
  left_hand: HandLandmark[] | null;       // 21 hand landmarks or null
  right_hand: HandLandmark[] | null;      // 21 hand landmarks or null
};

// ═══════════════════════════════════════════════════════════════
// AUDIO EVENTS
// ═══════════════════════════════════════════════════════════════

/**
 * Audio event (parent verbal prompt).
 */
export type AudioEvent = {
  type: string;                           // "CALL_ATTENTION" | "LOOK" etc.
  t_start_sec: number;
  t_end_sec: number;
  confidence: number;
  matched_phrase: string;                 // e.g., "look here", "hey buddy"
};

// ═══════════════════════════════════════════════════════════════
// EXPECTED WINDOWS
// ═══════════════════════════════════════════════════════════════

/**
 * Expected response window following a stimulus.
 */
export type ExpectedWindow = {
  type: string;                           // "orient_to_speaker" | "gaze_follow"
  trigger_type: string;                   // "CALL_ATTENTION" | "LOOK" | "POINT"
  trigger_phrase: string | null;
  t_start_sec: number;
  t_end_sec: number;
  expected_behavior: string;              // Human-readable description
  point_angle_deg?: number;               // For pointing events
};

// ═══════════════════════════════════════════════════════════════
// OVERLAY DATA (API Response)
// ═══════════════════════════════════════════════════════════════

/**
 * Complete overlay data for a video.
 * Returned by GET /videos/{video_id}/overlay-data
 */
export type OverlayData = {
  videoId: number;
  fps: number;
  durationMs: number;
  frames: OverlayFrame[];
  events: AudioEvent[];
  expectedWindows: ExpectedWindow[];
};

// ═══════════════════════════════════════════════════════════════
// ACTIVE EVENT STATE (for UI)
// ═══════════════════════════════════════════════════════════════

/**
 * Currently active event for display in the overlay.
 */
export type ActiveEvent = {
  event: AudioEvent;
  window: ExpectedWindow;
  status: "awaiting" | "monitoring" | "observed" | "not_observed";
  elapsedMs: number;
  windowDurationMs: number;
};

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

/**
 * Binary search to find the frame closest to a given time.
 */
export function findFrameAtTime(frames: OverlayFrame[], tSec: number): OverlayFrame | null {
  if (frames.length === 0) return null;

  let lo = 0;
  let hi = frames.length - 1;

  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (frames[mid].t_sec < tSec) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Return the closest frame (check both lo and lo-1)
  if (lo > 0) {
    const diffLo = Math.abs(frames[lo].t_sec - tSec);
    const diffPrev = Math.abs(frames[lo - 1].t_sec - tSec);
    if (diffPrev < diffLo) {
      return frames[lo - 1];
    }
  }

  return frames[lo];
}

/**
 * Find the active expected window for a given time.
 */
export function findActiveWindow(
  windows: ExpectedWindow[],
  tSec: number
): ExpectedWindow | null {
  for (const w of windows) {
    if (tSec >= w.t_start_sec && tSec <= w.t_end_sec) {
      return w;
    }
  }
  return null;
}

/**
 * Get display label for event type.
 */
export function getEventTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    "CALL_ATTENTION": "Name called",
    "LOOK": "Look phrase",
    "POINT": "Pointing gesture",
  };
  return labels[type] || type.replace("_", " ").toLowerCase();
}