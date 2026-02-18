import type { OverlayFrame } from "../../types/overlayTypes";
import { POSE_LANDMARKS } from "../../types/overlayTypes";

type Props = {
  frame: OverlayFrame | null;
  videoWidth: number;
  videoHeight: number;
  visible: boolean;
};

// Minimum confidence to show bounding box
const MIN_BBOX_CONFIDENCE = 0.3;

// Colors
const BBOX_COLOR = "rgba(100, 200, 255, 0.7)";
const ARROW_COLOR = "rgba(255, 200, 100, 0.9)";
const GAZE_CONE_COLOR = "rgba(100, 200, 255, 0.15)";

export default function HeadGazeOverlay({
  frame,
  videoWidth,
  videoHeight,
  visible,
}: Props) {
  if (!visible || !frame) return null;

  const { child_bbox, head_yaw, pose } = frame;

  // Calculate face center from pose if available (more accurate than bbox center)
  let faceCenterX: number | null = null;
  let faceCenterY: number | null = null;

  if (pose) {
    const nose = pose[POSE_LANDMARKS.NOSE];
    if (nose && nose[3] > 0.3) {
      faceCenterX = nose[0] * videoWidth;
      faceCenterY = nose[1] * videoHeight;
    }
  }

  // Fallback to bbox center
  if (faceCenterX === null && child_bbox && child_bbox[4] >= MIN_BBOX_CONFIDENCE) {
    const [x, y, w, h] = child_bbox;
    faceCenterX = (x + w / 2) * videoWidth;
    faceCenterY = (y + h / 2) * videoHeight;
  }

  return (
    <svg
      width={videoWidth}
      height={videoHeight}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    >
      {/* Arrow marker definition */}
      <defs>
        <marker
          id="gaze-arrowhead"
          markerWidth="8"
          markerHeight="6"
          refX="7"
          refY="3"
          orient="auto"
        >
          <polygon
            points="0 0, 8 3, 0 6"
            fill={ARROW_COLOR}
          />
        </marker>
        {/* Gradient for gaze cone */}
        <radialGradient id="gaze-cone-gradient" cx="0%" cy="50%" r="100%">
          <stop offset="0%" stopColor="rgba(100, 200, 255, 0.3)" />
          <stop offset="100%" stopColor="rgba(100, 200, 255, 0)" />
        </radialGradient>
      </defs>

      {/* Face bounding box */}
      {child_bbox && child_bbox[4] >= MIN_BBOX_CONFIDENCE && (
        <BoundingBox
          bbox={child_bbox}
          videoWidth={videoWidth}
          videoHeight={videoHeight}
        />
      )}

      {/* Head orientation arrow and gaze cone */}
      {faceCenterX !== null && faceCenterY !== null && head_yaw !== null && (
        <HeadOrientationIndicator
          centerX={faceCenterX}
          centerY={faceCenterY}
          headYaw={head_yaw}
        />
      )}
    </svg>
  );
}

function BoundingBox({
  bbox,
  videoWidth,
  videoHeight,
}: {
  bbox: number[];
  videoWidth: number;
  videoHeight: number;
}) {
  const [x, y, w, h, conf] = bbox;

  // Convert normalized coordinates to pixel coordinates
  const boxX = x * videoWidth;
  const boxY = y * videoHeight;
  const boxW = w * videoWidth;
  const boxH = h * videoHeight;

  // Opacity based on confidence
  const opacity = Math.min(1, conf + 0.3);

  return (
    <g opacity={opacity}>
      {/* Main bounding box */}
      <rect
        x={boxX}
        y={boxY}
        width={boxW}
        height={boxH}
        fill="none"
        stroke={BBOX_COLOR}
        strokeWidth={2}
        strokeDasharray="6,3"
        rx={4}
        ry={4}
      />
      {/* Corner accents for better visibility */}
      <CornerAccents
        x={boxX}
        y={boxY}
        width={boxW}
        height={boxH}
        color={BBOX_COLOR}
      />
    </g>
  );
}

function CornerAccents({
  x,
  y,
  width,
  height,
  color,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
}) {
  const cornerSize = Math.min(12, width * 0.15, height * 0.15);

  return (
    <g stroke={color} strokeWidth={3} strokeLinecap="round" fill="none">
      {/* Top-left */}
      <path d={`M ${x} ${y + cornerSize} L ${x} ${y} L ${x + cornerSize} ${y}`} />
      {/* Top-right */}
      <path d={`M ${x + width - cornerSize} ${y} L ${x + width} ${y} L ${x + width} ${y + cornerSize}`} />
      {/* Bottom-left */}
      <path d={`M ${x} ${y + height - cornerSize} L ${x} ${y + height} L ${x + cornerSize} ${y + height}`} />
      {/* Bottom-right */}
      <path d={`M ${x + width - cornerSize} ${y + height} L ${x + width} ${y + height} L ${x + width} ${y + height - cornerSize}`} />
    </g>
  );
}

function HeadOrientationIndicator({
  centerX,
  centerY,
  headYaw,
}: {
  centerX: number;
  centerY: number;
  headYaw: number;
}) {
  // Convert yaw to radians (yaw is in degrees, - = left, + = right)
  const yawRad = (headYaw * Math.PI) / 180;

  // Arrow length scales with yaw magnitude
  const arrowLength = Math.min(50, 20 + Math.abs(headYaw) * 0.8);

  // Arrow end point
  const endX = centerX + Math.sin(yawRad) * arrowLength;
  const endY = centerY - Math.cos(yawRad) * arrowLength * 0.6; // Flatten vertically

  // Only show if there's meaningful head rotation
  if (Math.abs(headYaw) < 5) {
    return (
      // Small dot when facing forward
      <circle
        cx={centerX}
        cy={centerY}
        r={4}
        fill={ARROW_COLOR}
        opacity={0.6}
      />
    );
  }

  // Gaze cone parameters
  const coneLength = 60;
  const coneSpread = 25; // degrees

  const coneLeftRad = yawRad - (coneSpread * Math.PI) / 180;
  const coneRightRad = yawRad + (coneSpread * Math.PI) / 180;

  const coneLeftX = centerX + Math.sin(coneLeftRad) * coneLength;
  const coneLeftY = centerY - Math.cos(coneLeftRad) * coneLength * 0.6;
  const coneRightX = centerX + Math.sin(coneRightRad) * coneLength;
  const coneRightY = centerY - Math.cos(coneRightRad) * coneLength * 0.6;

  return (
    <g>
      {/* Gaze cone (approximate attention direction) */}
      <path
        d={`M ${centerX} ${centerY} L ${coneLeftX} ${coneLeftY} A ${coneLength} ${coneLength * 0.6} 0 0 1 ${coneRightX} ${coneRightY} Z`}
        fill={GAZE_CONE_COLOR}
      />

      {/* Direction arrow */}
      <line
        x1={centerX}
        y1={centerY}
        x2={endX}
        y2={endY}
        stroke={ARROW_COLOR}
        strokeWidth={3}
        strokeLinecap="round"
        markerEnd="url(#gaze-arrowhead)"
      />

      {/* Center dot */}
      <circle
        cx={centerX}
        cy={centerY}
        r={4}
        fill={ARROW_COLOR}
      />

      {/* Yaw label */}
      <text
        x={centerX + 8}
        y={centerY - 8}
        fill={ARROW_COLOR}
        fontSize={11}
        fontFamily="system-ui, sans-serif"
        fontWeight={500}
      >
        {headYaw > 0 ? "+" : ""}{headYaw.toFixed(0)}°
      </text>
    </g>
  );
}