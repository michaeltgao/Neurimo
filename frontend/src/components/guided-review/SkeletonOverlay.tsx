import { useRef, useEffect } from "react";
import type { OverlayFrame, PoseLandmark } from "../../types/overlayTypes";
import { SKELETON_CONNECTIONS, POSE_LANDMARKS } from "../../types/overlayTypes";

type Props = {
  frame: OverlayFrame | null;
  videoWidth: number;
  videoHeight: number;
  visible: boolean;
};

// Visibility threshold for drawing landmarks
const MIN_VISIBILITY = 0.3;

// Colors
const LANDMARK_COLOR = "rgba(0, 255, 128, 0.8)";
const CONNECTION_COLOR = "rgba(0, 255, 128, 0.5)";
const FACE_COLOR = "rgba(100, 200, 255, 0.7)";

// Sizes
const LANDMARK_RADIUS = 3;
const CONNECTION_WIDTH = 2;

export default function SkeletonOverlay({
  frame,
  videoWidth,
  videoHeight,
  visible,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Don't draw if not visible or no frame data
    if (!visible || !frame?.pose) return;

    const pose = frame.pose;

    // Draw connections first (so landmarks appear on top)
    ctx.strokeStyle = CONNECTION_COLOR;
    ctx.lineWidth = CONNECTION_WIDTH;
    ctx.lineCap = "round";

    for (const [fromIdx, toIdx] of SKELETON_CONNECTIONS) {
      const from = pose[fromIdx];
      const to = pose[toIdx];

      if (!from || !to) continue;
      if (from[3] < MIN_VISIBILITY || to[3] < MIN_VISIBILITY) continue;

      const fromX = from[0] * videoWidth;
      const fromY = from[1] * videoHeight;
      const toX = to[0] * videoWidth;
      const toY = to[1] * videoHeight;

      // Set opacity based on average visibility
      const avgVisibility = (from[3] + to[3]) / 2;
      ctx.globalAlpha = avgVisibility;

      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    // Draw landmarks
    for (let i = 0; i < pose.length; i++) {
      const landmark = pose[i];
      if (!landmark || landmark[3] < MIN_VISIBILITY) continue;

      const x = landmark[0] * videoWidth;
      const y = landmark[1] * videoHeight;
      const visibility = landmark[3];

      // Use different color for face landmarks
      const isFaceLandmark = i <= POSE_LANDMARKS.MOUTH_RIGHT;
      ctx.fillStyle = isFaceLandmark ? FACE_COLOR : LANDMARK_COLOR;
      ctx.globalAlpha = visibility;

      ctx.beginPath();
      ctx.arc(x, y, LANDMARK_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }, [frame, videoWidth, videoHeight, visible]);

  if (!visible) return null;

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}