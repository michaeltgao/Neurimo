import type { ExpectedWindow, AudioEvent } from "../../types/overlayTypes";
import { getEventTypeLabel } from "../../types/overlayTypes";

type Props = {
  currentTimeSec: number;
  events: AudioEvent[];
  expectedWindows: ExpectedWindow[];
  visible: boolean;
};

type ActiveWindowState = {
  window: ExpectedWindow;
  event: AudioEvent | null;
  status: "awaiting" | "monitoring" | "ended";
  elapsedSec: number;
  windowDurationSec: number;
  progressPct: number;
};

export default function EventCallout({
  currentTimeSec,
  events,
  expectedWindows,
  visible,
}: Props) {
  if (!visible) return null;

  // Find currently active window
  const activeWindow = findActiveWindow(currentTimeSec, expectedWindows, events);

  if (!activeWindow) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: 16,
        left: 16,
        right: 16,
        display: "flex",
        justifyContent: "center",
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.85)",
          backdropFilter: "blur(8px)",
          borderRadius: 12,
          padding: "12px 20px",
          maxWidth: 400,
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
        }}
      >
        {/* Stimulus label */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 8,
          }}
        >
          <span
            style={{
              fontSize: "1.25rem",
            }}
          >
            {getStatusEmoji(activeWindow.status)}
          </span>
          <span
            style={{
              color: "#fff",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            {activeWindow.event
              ? getEventTypeLabel(activeWindow.event.type)
              : activeWindow.window.trigger_type}
          </span>
          {activeWindow.event?.matched_phrase && (
            <span
              style={{
                color: "rgba(255, 255, 255, 0.6)",
                fontSize: "0.75rem",
                fontStyle: "italic",
              }}
            >
              "{activeWindow.event.matched_phrase}"
            </span>
          )}
        </div>

        {/* Expected behavior */}
        <div
          style={{
            color: "rgba(255, 255, 255, 0.7)",
            fontSize: "0.75rem",
            marginBottom: 10,
          }}
        >
          Expected: {activeWindow.window.expected_behavior}
        </div>

        {/* Progress bar */}
        <div
          style={{
            height: 6,
            backgroundColor: "rgba(255, 255, 255, 0.2)",
            borderRadius: 3,
            overflow: "hidden",
            marginBottom: 8,
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${activeWindow.progressPct}%`,
              backgroundColor: getProgressColor(activeWindow.status, activeWindow.progressPct),
              borderRadius: 3,
              transition: "width 0.1s linear, background-color 0.3s ease",
            }}
          />
        </div>

        {/* Time display */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span
            style={{
              color: "rgba(255, 255, 255, 0.5)",
              fontSize: "0.6875rem",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {activeWindow.elapsedSec.toFixed(1)}s / {activeWindow.windowDurationSec.toFixed(1)}s
          </span>
          <StatusChip status={activeWindow.status} />
        </div>
      </div>
    </div>
  );
}

function findActiveWindow(
  currentTimeSec: number,
  windows: ExpectedWindow[],
  events: AudioEvent[]
): ActiveWindowState | null {
  // Find window that is currently active or just ended (within 0.5s)
  for (const window of windows) {
    const windowStart = window.t_start_sec;
    const windowEnd = window.t_end_sec;
    const windowDuration = windowEnd - windowStart;

    // Window is active
    if (currentTimeSec >= windowStart && currentTimeSec <= windowEnd) {
      const elapsed = currentTimeSec - windowStart;
      const progress = (elapsed / windowDuration) * 100;

      // Find matching event
      const event = events.find(
        (e) => e.t_start_sec <= windowStart && e.t_end_sec >= windowStart - 0.5
      ) || events.find(
        (e) => Math.abs(e.t_start_sec - windowStart) < 1.0
      );

      return {
        window,
        event: event || null,
        status: "monitoring",
        elapsedSec: elapsed,
        windowDurationSec: windowDuration,
        progressPct: Math.min(100, progress),
      };
    }

    // Window just ended (show for 1 second after)
    if (currentTimeSec > windowEnd && currentTimeSec <= windowEnd + 1.0) {
      const event = events.find(
        (e) => Math.abs(e.t_start_sec - windowStart) < 1.0
      );

      return {
        window,
        event: event || null,
        status: "ended",
        elapsedSec: windowDuration,
        windowDurationSec: windowDuration,
        progressPct: 100,
      };
    }

    // Window about to start (show 0.5s before)
    if (currentTimeSec >= windowStart - 0.5 && currentTimeSec < windowStart) {
      const event = events.find(
        (e) => Math.abs(e.t_start_sec - windowStart) < 1.0
      );

      return {
        window,
        event: event || null,
        status: "awaiting",
        elapsedSec: 0,
        windowDurationSec: windowDuration,
        progressPct: 0,
      };
    }
  }

  return null;
}

function getStatusEmoji(status: "awaiting" | "monitoring" | "ended"): string {
  switch (status) {
    case "awaiting":
      return "🎯";
    case "monitoring":
      return "⏱";
    case "ended":
      return "📋";
  }
}

function getProgressColor(
  status: "awaiting" | "monitoring" | "ended",
  progressPct: number
): string {
  if (status === "ended") {
    return "rgba(100, 200, 255, 0.8)";
  }
  if (progressPct > 80) {
    return "rgba(255, 180, 100, 0.9)";
  }
  return "rgba(100, 255, 150, 0.8)";
}

function StatusChip({ status }: { status: "awaiting" | "monitoring" | "ended" }) {
  const config = {
    awaiting: {
      label: "Starting...",
      bg: "rgba(100, 200, 255, 0.2)",
      color: "rgba(100, 200, 255, 1)",
    },
    monitoring: {
      label: "Monitoring",
      bg: "rgba(100, 255, 150, 0.2)",
      color: "rgba(100, 255, 150, 1)",
    },
    ended: {
      label: "Window closed",
      bg: "rgba(255, 255, 255, 0.1)",
      color: "rgba(255, 255, 255, 0.6)",
    },
  };

  const { label, bg, color } = config[status];

  return (
    <span
      style={{
        padding: "3px 8px",
        backgroundColor: bg,
        color: color,
        borderRadius: 4,
        fontSize: "0.6875rem",
        fontWeight: 500,
      }}
    >
      {label}
    </span>
  );
}