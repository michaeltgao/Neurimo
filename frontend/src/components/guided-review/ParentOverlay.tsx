import { useRef, useEffect } from "react";
import type { OverlayFrame, HandLandmark } from "../../types/overlayTypes";
import { HAND_CONNECTIONS } from "../../types/overlayTypes";

type Props = {
  frame: OverlayFrame | null;
  videoWidth: number;
  videoHeight: number;
  visible: boolean;
};

// Colors for parent overlay (distinct from child which uses cyan/green)
const PARENT_BOX_COLOR = "#f97316"; // Orange
const HAND_COLOR = "#fbbf24"; // Amber/yellow
const HAND_JOINT_COLOR = "#fef3c7"; // Light amber

/**
 * Renders parent bounding box and hand landmarks on a canvas overlay.
 * Shows when parent is detected in frame.
 */
export default function ParentOverlay({
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

    if (!visible || !frame) return;

    const hasParent = frame.parent_bbox || frame.left_hand || frame.right_hand;
    if (!hasParent) return;

    // Draw parent bounding box
    if (frame.parent_bbox) {
      const [x, y, w, h, conf] = frame.parent_bbox;

      // Only draw if confidence is reasonable
      if (conf > 0.2) {
        const px = x * videoWidth;
        const py = y * videoHeight;
        const pw = w * videoWidth;
        const ph = h * videoHeight;

        ctx.strokeStyle = PARENT_BOX_COLOR;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]); // Dashed line to distinguish from child
        ctx.strokeRect(px, py, pw, ph);
        ctx.setLineDash([]); // Reset

        // Label
        ctx.fillStyle = PARENT_BOX_COLOR;
        ctx.font = "bold 12px system-ui";
        ctx.fillText("Parent", px + 4, py - 6);
      }
    }

    // Draw left hand
    if (frame.left_hand) {
      drawHand(ctx, frame.left_hand, videoWidth, videoHeight, "L");
    }

    // Draw right hand
    if (frame.right_hand) {
      drawHand(ctx, frame.right_hand, videoWidth, videoHeight, "R");
    }
  }, [frame, videoWidth, videoHeight, visible]);

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

/**
 * Draw hand landmarks and connections.
 */
function drawHand(
  ctx: CanvasRenderingContext2D,
  landmarks: HandLandmark[],
  videoWidth: number,
  videoHeight: number,
  label: string
) {
  if (landmarks.length !== 21) return;

  // Check if hand has valid data (not all NaN)
  const validLandmarks = landmarks.filter(
    ([x, y, , vis]) =>
      !isNaN(x) && !isNaN(y) && vis > 0.3
  );

  if (validLandmarks.length < 5) return; // Need at least 5 visible points

  // Draw connections first (under joints)
  ctx.strokeStyle = HAND_COLOR;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.8;

  for (const [from, to] of HAND_CONNECTIONS) {
    const [x1, y1, , vis1] = landmarks[from];
    const [x2, y2, , vis2] = landmarks[to];

    if (vis1 > 0.3 && vis2 > 0.3 && !isNaN(x1) && !isNaN(x2)) {
      ctx.beginPath();
      ctx.moveTo(x1 * videoWidth, y1 * videoHeight);
      ctx.lineTo(x2 * videoWidth, y2 * videoHeight);
      ctx.stroke();
    }
  }

  // Draw joints
  ctx.fillStyle = HAND_JOINT_COLOR;
  ctx.globalAlpha = 1;

  for (const [x, y, , vis] of landmarks) {
    if (vis > 0.3 && !isNaN(x) && !isNaN(y)) {
      ctx.beginPath();
      ctx.arc(x * videoWidth, y * videoHeight, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Draw label at wrist
  const [wristX, wristY, , wristVis] = landmarks[0];
  if (wristVis > 0.3 && !isNaN(wristX)) {
    ctx.fillStyle = HAND_COLOR;
    ctx.font = "bold 10px system-ui";
    ctx.fillText(
      label === "L" ? "L" : "R",
      wristX * videoWidth - 12,
      wristY * videoHeight - 8
    );
  }

  ctx.globalAlpha = 1;
}